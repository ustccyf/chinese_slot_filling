import sys,os
import numpy as np
import tensorflow as tf

def load_vocab(filename):
    d = {}
    reverse_d = {}
    idx = 0
    with open(filename) as f:
        for line in f:
            w = line.strip()
            if w in d:
                continue
            d[w] = idx
            reverse_d[idx] = w
            idx += 1
    sys.stderr.write('load vocab from file:[%s], len(d):[%d], len(revers_d):[%d]\n' \
            % (filename, len(d), len(reverse_d)))
    return d, reverse_d

def get_trimmed_glove_vectors(filename, vocab_words, dim_word):
    #return matrix of embeddings
    #with np.load(filename) as data:
    embeddings = np.random.rand(len(vocab_words), dim_word)
    fin = open(filename, "r")
    for line in fin:
        lin = line.strip().split(" ")
        if len(lin) != 201:
            continue
        if lin[0] in vocab_words:
            idx = vocab_words[lin[0]]
            emb = [float(x) for x in lin[1:]]
            embeddings[idx] = np.asarray(emb)
    fin.close()
    return embeddings
#data = np.load(filename)
#return data["embeddings"]

class Config():
    def __init__(self, domain, use_emb=0):
        #self.use_pretrained = False
        self.use_pretrained = True
        if use_emb == -1:
            self.use_pretrained = False
        """
        self.vocabfile = "data/lbs/words.txt"
        self.tagfile = "data/lbs/output_slot.txt"
        self.intentfile = "data/lbs/intent.txt"
        self.inputtagfile = "data/lbs/input_slot.txt"
        self.trainfile = "data/lbs/train.txt"
        self.devfile = "data/lbs/dev.txt"
        self.testfile = "data/lbs/test.txt"
        self.dir_output = "results/lbs/"
        self.dir_model = self.dir_output + "model.weights/"
        self.filename_trimmed = "data/lbs/word.{}d.trimmed.npz".format(self.dim_word)
        """
        self.vocabfile = "data/" + domain + "/words.txt"
        self.tagfile = "data/" + domain + "/output_slot.txt"
        self.intentfile = "data/" + domain + "/intent.txt"
        self.inputtagfile = "data/" + domain + "/input_slot.txt"
        self.trainfile = "data/" + domain + "/train.txt"
        self.devfile = "data/" + domain + "/dev.txt"
        self.testfile = "data/" + domain  + "/test.txt"
        self.dir_output = "results/" + domain + "/"
        self.dir_model = self.dir_output + "model.weights/"
        self.dim_word = 200
        self.filename_trimmed = "data/" + domain + \
                "/word.{}d.trimmed.npz".format(self.dim_word)
        #self.vocabfile = "data/music.vocab"
        #self.tagfile = "data/musictag.vocab"
        #self.trainfile = "data/music.train"
        #self.devfile = "data/music.dev"
        #self.testfile = "data/music.test"
        self.nepochs = 150
        #self.nepochs = 50
        self.log_file = "log/logger"
        self.batch_size = 10
        self.lr = 0.005
        self.intent_weight = 0.5
        self.dropout = 0.8
        self.lr_decay = 0.9
        self.nepoch_no_imprv  = 10
        self.vocab_words, self.idx2vocab = load_vocab(self.vocabfile)
        #for item in self.vocab_words:
        #    print item + "\t" + str(self.vocab_words[item])
        self.vocab_tags, self.idx2tag = load_vocab(self.tagfile)
        self.vocab_intents, self.idx2intent = load_vocab(self.intentfile)
        self.vocab_inputtags, self.idx2inputintent = \
                load_vocab(self.inputtagfile)
        self.vocab_size = len(self.vocab_words)
        self.tag_size = len(self.vocab_tags)
        self.num_intents = len(self.vocab_intents)
        self.inputtag_size = len(self.vocab_inputtags)
        self.mlp_num_units = 256
        self.hidden_size_lstm = 50
        self.attention_size = 50
        #self.hidden_size_lstm = 200
        #self.filename_trimmed = "data/glove.6B.{}d.trimmed.npz".format(300)
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed, self.vocab_words, self.dim_word) if self.use_pretrained else None)
        self.lr_method = "adam"
        self.clip = -1 # if negative, no clipping

        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[2] == "emb":
        config = Config(sys.argv[1] ,-1)
        embeddings = np.random.rand(len(config.vocab_words), config.dim_word)
        #fin = open("/home/work/chenyanfeng/dict/tencent.emb.cache", "r")
        #fin = open("dict/flight.input.word.emb", "r")
        fin = open("../embedding/Tencent_AILab_ChineseEmbedding.txt", "r")
        fout = open(config.filename_trimmed, "w")
        for line in fin:
            lin = line.strip().split(" ")
            if len(lin) != 201:
                continue
            if lin[0] in config.vocab_words:
                fout.write(line)
            #if lin[0] in config.vocab_words:
            #    idx = config.vocab_words[lin[0]]
            #    emb = [float(x) for x in lin[1:]]
            #    embeddings[idx] = np.asarray(emb)
        fin.close()
        fout.close()
        #np.savez_compressed(config.filename_trimmed, embeddings=embeddings)
