#coding=utf8
import sys,os
import tensorflow as tf
import logging
import numpy as np
from config import Config
from data_utils import VocabDict, minibatches,get_chunks
from general_utils import Progbar

class SlotFillingModel():
    def __init__(self, config, domain):
        self.config = config
        self.domain = "_" + domain
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'gpu': 0}))
        self.word_ids = tf.placeholder(tf.int32, shape = [None, None],
                name = "word_ids" + self.domain)
        self.sequence_lengths = tf.placeholder(tf.int32, shape = [None],
                name = "sequence_lengths" + self.domain)
        #self.input_labels = tf.placeholder(tf.int32, shape = [None, None, None],
        self.input_labels = tf.placeholder(tf.int32, shape = [None, None],
                name = "input_labels" + self.domain)
        self.labels = tf.placeholder(tf.int32, shape = [None, None],
                name = "labels" + self.domain)
        self.able_labels = tf.placeholder(tf.float32, shape = [None, None, self.config.inputtag_size], name = "able_labels" + self.domain)
        self.intent = tf.placeholder(tf.int32, shape=[None], name="intent"\
                + self.domain)
        self.dropout = tf.placeholder(tf.float32, shape = [],
                name = "dropout" + self.domain)
        self.lr = tf.placeholder(tf.float32, shape = [], \
                name = "lr" + self.domain)
        self.logger = self.get_logger()
        self.saver = None

    def get_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        handler = logging.FileHandler(self.config.log_file)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        return logger

    def add_word_embeddings_op(self, vocab_size, dim_word):
        with tf.variable_scope("emb_layer" + self.domain):
            if self.config.embeddings is None:
                embed_matrix = tf.get_variable(
                        name="_word_embeddings" + self.domain,
                        dtype=tf.float32,
                        shape=[vocab_size, dim_word])
                self.embedding_trainable = True
            else:
                embed_matrix = tf.Variable(
                        self.config.embeddings,
                        name = "_word_embeddings" + self.domain,
                        dtype = tf.float32,
                        trainable = False)
                self.embedding_trainable = False
            word_embeddings = tf.nn.embedding_lookup(embed_matrix,
                    self.word_ids, name = "word_embedding" + self.domain)
            #able_labels = tf.one_hot(self.input_labels, self.config.tag_size, on_value=1., off_value=0.)
            #merge_embeddings = word_embeddings
            self.merge_embeddings = word_embeddings
            #merge_embeddings = tf.concat([word_embeddings, self.able_labels], -1)

            self.word_embeddings = tf.nn.dropout(self.merge_embeddings, \
                    self.dropout, name="word_embeddings" + self.domain)
        #self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)
    
    def set_word_embeddings_trainable(self):
        self.embedding_trainable = True
        with tf.variable_scope("emb_layer" + self.domain):
            embed_matrix = tf.Variable(
                    self.config.embeddings,
                    name="_word_embeddings" + self.domain,
                    dtype=tf.float32,
                    trainable=True)
            word_embeddings = tf.nn.embedding_lookup(embed_matrix, self.word_ids, name="word_embedding"+ self.domain)
            #merge_embeddings = tf.concat([word_embeddings, self.label_embeddings], -1, name="concat_embedding_tag")
            self.merge_embeddings = word_embeddings
            self.word_embeddings = tf.nn.dropout(self.merge_embeddings, self.dropout, name="word_embeddings"+ self.domain)

    def add_birnn(self, hidden_size_lstm):
        with tf.variable_scope("bi-lstm" + self.domain):
            cell_fw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length = self.sequence_lengths,
                    dtype=tf.float32, scope="bi-lstm" + self.domain)
            output = tf.concat([output_fw, output_bw], axis=-1,\
                    name = "lstm_concat" + self.domain)
            lstm_output = tf.nn.dropout(output, self.dropout, \
                    name = "lstm_output" + self.domain)
            #merge_embeddings = tf.concat([word_embeddings, self.able_labels], -1)

            #self.lstm_outputs = tf.nn.dropout(output, self.dropout)
            self.lstm_outputs = tf.concat([lstm_output, self.able_labels],\
                    -1, name = "lstm_output_concat")

    def add_slot_proj(self, hidden_size_lstm, ntags):
        with tf.variable_scope("proj" + self.domain):
            W = tf.get_variable(name = "W" + self.domain, dtype = tf.float32,
                    shape = [2 * hidden_size_lstm + self.config.inputtag_size, ntags])
            b = tf.get_variable(name = "b" + self.domain, shape = [ntags],
                    dtype = tf.float32, initializer=tf.zeros_initializer())
            nsteps = tf.shape(self.lstm_outputs, \
                    name = "nsteps" + self.domain)[1] #sequence(每句话)的长度
            output = tf.reshape(self.lstm_outputs, [-1, 2 * hidden_size_lstm + self.config.inputtag_size],
                    name = "output" + self.domain)
#reshape [ , , 2*hidden_size_lstm]到这个,原因是因为要做一个映射，输入是词向量，输出是slot
            pred = tf.matmul(output, W, name = "pred" + self.domain) + b
            self.logits = tf.reshape(pred, [-1, nsteps, ntags], name = "logits" + self.domain)
            #shape为[个数、sequences长度、tag个数]
        return

    def add_attention_layer(self, hidden_size_lstm):
        with tf.variable_scope("attention" + self.domain):
            #学习一个向量v_att，对lstm输出点乘之后softmax之后对所有lstm输出进行加权,得到最终的query的表示
            nsteps = tf.shape(self.lstm_outputs, name = "att_nsteps" + self.domain)[1] #sequence(每句话)的长度
            max_time = tf.shape(self.lstm_outputs, name = "max_time")[1] #sequence(每句话)的长度
            output = tf.reshape(self.lstm_outputs, [-1, 2 * hidden_size_lstm \
                    + self.config.inputtag_size], name = "output_att")
            combined_hidden_size = 2 * hidden_size_lstm + self.config.inputtag_size
            atten_size = self.config.attention_size
            #atten_size = 2 * hidden_size_lstm
            W_omega = tf.Variable(tf.random_normal(\
                    [combined_hidden_size, atten_size], stddev = 0.1,\
                    dtype = tf.float32), name = "w_omega" + self.domain)
            b_omega = tf.Variable(tf.random_normal(\
                    [atten_size], stddev = 0.1,\
                    dtype = tf.float32), name= "b_omega" + self.domain)
            u_omega = tf.Variable(tf.random_normal(\
                    [atten_size], stddev = 0.1, dtype = tf.float32),\
                    name = "u_omega" + self.domain)
            """
            W_omega = tf.get_variable("W", dtype = tf.float32,
                    shape = [2 * hidden_size_lstm, 2 * hidden_size_lstm])
            b_omega = tf.get_variable("b", shape = [2 * hidden_size_lstm],
                    dtype = tf.float32, initializer=tf.zeros_initializer())
            u_omega = tf.get_variable("u", shape = [2 * hidden_size_lstm],
                    dtype = tf.float32, initializer=tf.zeros_initializer())
            """
            pred = tf.matmul(output, W_omega) + tf.reshape(b_omega, [1, -1])
            v = tf.tanh(pred, name = "v" + self.domain)
            vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]), name="vu" + self.domain)
            exps = tf.reshape(tf.exp(vu), [-1, max_time], name = "exps" + self.domain)
            alphas = exps/ tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
            atten_outs = tf.reduce_sum(self.lstm_outputs * \
                    tf.reshape(alphas, [-1, max_time, 1]), 1)
            #logits = tf.reshape(pred, [-1, nsteps, ntags])
            #v_att = tf.get_variable("v_att", shape=[hidden_size_lstm * 2],dtype=tf.float32)
            #scores = tf.reduce_sum(v_att * self.lstm_outputs, [2])
            #nums_scores = tf.shape(scores)[1]
            #scores_mask = tf.sequence_mask(
            #        lengths=tf.to_int32(self.sequence_lengths),
             #       maxlen = tf.to_int32(nums_scores),
            #        dtype=tf.float32)
            #scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)
            #scores_normalized = tf.nn.softmax(scores)
            #context = tf.expand_dims(scores_normalized, 2) * self.lstm_outputs
            #context = tf.reduce_sum(context, 1)
            #context.set_shape([None, hidden_size_lstm * 2])
            #self.context = context
            self.context = atten_outs

    def add_intent_mlp(self):
        with tf.variable_scope("intent_mlp" + self.domain):
            self.mlp_output = tf.contrib.layers.fully_connected(
                    inputs=self.context,
                    num_outputs = self.config.mlp_num_units,
                    activation_fn=tf.nn.relu,
                    biases_initializer=tf.random_uniform_initializer(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                    scope="intent_mlp" + self.domain)
        with tf.variable_scope("intent_logits" + self.domain):
            self.intent_logits = tf.contrib.layers.fully_connected(
                    inputs=self.mlp_output,
                    num_outputs=self.config.num_intents,
                    activation_fn=None,
                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                    biases_initializer=tf.random_uniform_initializer(),
                    scope="intent_logits" + self.domain)
            self.y = tf.argmax(self.intent_logits, 1)
            self.score = tf.nn.softmax(self.intent_logits)
    def add_intent_loss(self):
        with tf.variable_scope("intent_loss" + self.domain):
            label = tf.one_hot(self.intent, self.config.num_intents,
                    on_value=1, off_value=0)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                    labels=label,logits=self.intent_logits)
            self.intent_loss = tf.reduce_mean(losses)
        #with tf.variable_scope("logits_intent"):
        #    lstm_outputs_reshape = tf.reshape(lstm_outputs, [-1, self.nsteps, 2* hidden_size_lstm])
        #    lstm_outputs_reshape = lstm_outputs_reshape[:, -1, :]#取lstm最后一个
        #    W = tf.get_variable("W", shape=[hidden_size_lstm * 2, self.
        #        config.num_intents],dtype=tf.float32, initializer=self.initializer)
        #    b = tf.get_variable("b", shape=[self.config.num_intents], dtype=tf.float32,
        #            initializer=tf.zeros_initializer())
        #    pred_intent = tf.nn.xw_plus_b(lstm_outputs_reshape, W, b)
    #def add_predict_op(self):
    #    self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
    #            tf.int32)
    def add_loss_op(self):
        with tf.variable_scope("loss_op" + self.domain):
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.slot_loss = tf.reduce_mean(-log_likelihood)
            #self.loss = self.slot_loss + self.intent_loss
            self.loss = (1-self.config.intent_weight) *self.slot_loss \
                    + self.config.intent_weight * self.intent_loss
            tf.summary.scalar("loss", self.loss)
    def add_train_op(self, lr_method, lr, loss, clip=-1):
        _lr_m = lr_method.lower()
        with tf.variable_scope("train_step" + self.domain):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lr, name="optimizer"\
                        + self.domain)
                #optimizer_emb = tf.train.AdamOptimizer(lr * 0.001, name="optimizer_emb"\
                #        + self.domain)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr, name="optimizer"\
                        + self.domain)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr,
                        name="optimizer" + self.domain)
            else:
                optimizer = tf.train.AdamOptimizer(lr)
            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                #self.train_op_other = optimizer.minimize(loss)
                #self.train_op_emb = optimizer_emb.minimize(loss, \
                #        var_list=self.merge_embeddings)
                #self.train_op = tf.group(train_op_emb, train_op_other)
                self.train_op = optimizer.minimize(loss)

    def initialize_session(self):
        self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def add_summary(self):
        self.merged      = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output,
                self.sess.graph)

    def pad_sequences(self, words , pad_tok):
        sequence_padded, sequence_length = [], []
        max_length = max(map(lambda x:len(x), words))
        for seq in words:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
            sequence_padded +=  [seq_]
            sequence_length += [min(len(seq), max_length)]
        return sequence_padded, sequence_length

    def get_feed_dict(self, words, input_labels, labels=None, intent=None, lr=None, dropout=None):
        sequence_padded, sequence_lengths = self.pad_sequences(words, 0)
        input_labels_padded, _ = self.pad_sequences(input_labels, [0]*self.config.inputtag_size)
        #onehot_tag = tf.one_hot(input_labels_padded, self.config.tag_size, on_value=1., off_value=0.)
        #sequence_input_labels, _ = self.pad_sequences(input_labels, 0)
        feed = {
                self.word_ids: sequence_padded,
                self.sequence_lengths: sequence_lengths,
                self.able_labels: input_labels_padded,
                }
        if intent is not None:
            feed[self.intent] = intent
        if labels is not None:
            labels, _ = self.pad_sequences(labels, 0)
            feed[self.labels] = labels
        if lr is not None:
            feed[self.lr] = lr
        if dropout is not None:
            feed[self.dropout] = dropout
        return feed, sequence_lengths

    def save_session(self):
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

    def restore_session(self, dir_model):
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def predict_batch(self, words, labels):
        #输入：sentences的list
        #输出：labels_pred: label的list
        #       sequence_length
        fd, sequence_lengths = self.get_feed_dict(words, labels, dropout=1.0)
        viterbi_sequences = []
        y, score, logits, trans_params = self.sess.run(
                [self.y, self.score, self.logits, self.trans_params], feed_dict=fd)
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length] # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
            viterbi_sequences += [viterbi_seq]
        return viterbi_sequences, sequence_lengths, y, score
    
    def predict_once(self, words, input_tags):
        words = [words]
        input_tags = [input_tags]
        labels_pred, sequence_lengths, pred_intents, score = self.predict_batch(words, input_tags)
        return labels_pred, sequence_lengths, pred_intents, score

    def run_evaluate(self, test, print_or_not = False):
        accs = []
        intent_correct = 0
        intent_total = 0
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels, intents, all_tags in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths, pred_intents, score = self.predict_batch(words, all_tags)
            for word_ins, lab, lab_pred, length, intent, pred_intent in\
                    zip(words, labels, labels_pred,
                    sequence_lengths, intents, pred_intents):
                if print_or_not:
                    #words_list = [str(a) for a in words_ins]
                    #lab_list = [str(a) for a in lab]
                    #lab_pred_list = [str(a) for a in lab_pred ]
                    words_list = [self.config.idx2vocab[a] for a in word_ins]
                    lab_list = [self.config.idx2tag[a] for a in lab]
                    lab_pred_list = [self.config.idx2tag[a] for a in lab_pred ]
                    print "||".join(words_list) + "\t" + "||".join(lab_list) \
                            + "\t" + "||".join(lab_pred_list) + "\t" \
                            + str(self.config.idx2intent[intent]) + "\t"\
                            + str(self.config.idx2intent[pred_intent])
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a==b for (a,b) in zip(lab, lab_pred)]
                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,self.config.vocab_tags))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)
                intent_total += 1
                if pred_intent == intent:
                    intent_correct += 1
        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        if intent_total != 0:
            intent_acc = intent_correct / float(intent_total)
        else:
            intent_acc = 0
        return {"acc": 100*acc, "f1": 100*f1, "intent_acc": 100* intent_acc, \
                "intent_correct": intent_correct, "intent_total": intent_total}

    def evaluate(self, test, predict_or_not = False):
        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test, predict_or_not)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
            for k, v in metrics.items()])
        print msg
        #self.logger.info(msg)

    def train(self, train, dev):
        best_score = 0
        nepoch_no_imprv = 0 # for early stopping
        self.add_summary() # tensorboard

        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                self.config.nepochs))
            batch_size = self.config.batch_size
            nbatches = (len(train) + batch_size - 1) // batch_size
            prog = Progbar(target=nbatches)
            #self.config.lr *= self.config.lr_decay
            for i, (words, labels, intent, all_tags) in enumerate(minibatches(train, batch_size)):
                fd, _ = self.get_feed_dict(words, all_tags, labels, intent, self.config.lr,\
                        self.config.dropout)
                _, train_loss, summary, intent_loss, slot_loss= self.sess.run(
                        [self.train_op, self.loss, self.merged, self.intent_loss, self.slot_loss], feed_dict=fd)
                prog.update(i + 1, [("train loss", train_loss), \
                        ("intent_loss", intent_loss), ("slot_loss", slot_loss)])
                if i % 10 == 0:
                    self.file_writer.add_summary(summary, epoch*nbatches + i)
            metrics = self.run_evaluate(dev)
            msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
            self.logger.info(msg)
            score = metrics["f1"] + metrics["intent_acc"]
            self.config.lr *= self.config.lr_decay
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    if not self.embedding_trainable:
                        self.logger.info("fine tuning word embedding")
                        for i in range(10):
                            self.logger.info("######################")
                        self.set_word_embeddings_trainable()
                        self.config.lr = 0.001
                        nepoch_no_imprv = 0
                    else:
                        self.logger.info("- early stopping {} epochs without "\
                                "improvement".format(nepoch_no_imprv))
                        break

    def build(self, vocab_size, dim_word, hidden_size_lstm, ntags):
        self.add_word_embeddings_op(vocab_size, dim_word)
        self.add_birnn(hidden_size_lstm)
        self.add_attention_layer(hidden_size_lstm)
        self.add_slot_proj(hidden_size_lstm, ntags)
        self.add_intent_mlp()
        self.add_intent_loss()
        self.add_loss_op()
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session()

if __name__ == "__main__":
    config = Config(sys.argv[1])
    train_file = config.trainfile
    dev_file = config.devfile
    vocabdict = VocabDict(config)
    train = vocabdict.load_flight_data(config.trainfile)
    dev = vocabdict.load_flight_data(config.devfile)
    model = SlotFillingModel(config, sys.argv[1])
    model.build(config.vocab_size, config.dim_word, config.hidden_size_lstm,\
            config.tag_size)
    model.train(train, dev)
