import sys,os
import numpy as np
from config import Config
UNK = "$UNK$"
NUM = "$NUM$"
TAG_UNK = "O"
NONE = "O"

def minibatches(data, minibatch_size):
    x_batch, y_batch, z_batch, o_batch = [], [], [],[]
    for (x, y, z, o) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch, z_batch, o_batch
            x_batch, y_batch, z_batch, o_batch = [], [], [], []
        x_batch += [x]
        y_batch += [y]
        z_batch.append(z)
        o_batch += [o]
    if len(x_batch) != 0:
        yield x_batch, y_batch, z_batch, o_batch

class VocabDict:
    def __init__(self, config):
        self.config = config
        self.word_dict = config.vocab_words
        self.inputtag_dict = config.vocab_inputtags
        self.tag_dict = config.vocab_tags
        self.intent_dict = config.vocab_intents
    def load_music_data(self, train_filename):
        fin = open(train_filename, "r")
        result = []
        words = []
        tags = []
        for line in fin:
            word_ins = []
            tag_ins = []
            all_tag_ins = []
            lin = line.strip().split("\t")
            if len(lin) != 3:
                continue
            intent = self.intent_dict[lin[2]]
            for item in lin[0].split("||"):
                item = item.lower()
                if item.isdigit():
                    word = NUM
                if item in self.word_dict:
                    word_ins.append(self.word_dict[item])
                else:
                    word_ins.append(self.word_dict[UNK])
            for item in lin[1].split("||"):
                if item in self.tag_dict:
                    tag_ins.append(self.tag_dict[item])
                else:
                    tag_ins.append(self.tag_dict[UNK])
            for item in lin[1].split("||"):
                item_vec = [0] * self.config.tag_size
                if item in self.tag_dict:
                    item_vec[self.tag_dict[item]] = 1
                else:
                    item_vec[self.tag_dict[UNK]] = 1
                all_tag_ins.append(item_vec)
            if len(word_ins) != len(tag_ins):
                print >>sys.stderr, line.strip()
                continue
            result.append([word_ins, tag_ins, intent, all_tag_ins])
            words.append(word_ins)
            tags.append(tag_ins)
        fin.close()
        return result
        #return words,tags
    def load_flight_data(self, train_filename):
        fin = open(train_filename, "r")
        result = []
        words = []
        tags = []
        input_tags = []
        for line in fin:
            word_ins = []
            tag_ins = []
            all_tag_ins = []
            input_tag_ins = []
            lin = line.strip().split("\t")
            if len(lin) != 4:
                continue
            intent = self.intent_dict[lin[2]]
            for idx,item in enumerate(lin[0].split("||")):
                item = item.lower()
                if lin[3].split("||")[idx].find("TIME") != -1:
                    item = "$TIME$"
                elif item.isdigit():
                    item = NUM
                if item in self.word_dict:
                    word_ins.append(self.word_dict[item])
                else:
                    word_ins.append(self.word_dict[UNK])
            for item in lin[1].split("||"):
                if item in self.tag_dict:
                    tag_ins.append(self.tag_dict[item])
                else:
                    tag_ins.append(self.tag_dict[TAG_UNK])
            for item in lin[3].split("||"):
                item_vec = [0] * self.config.inputtag_size
                if item != "O":
                    for subtag in item.split("\x01"):
                        if subtag in self.inputtag_dict:
                            item_vec[self.inputtag_dict[subtag]] = 1
                all_tag_ins.append(item_vec)
            if len(word_ins) != len(tag_ins):
                print >>sys.stderr, line.strip()
                continue
            if len(word_ins) and len(tag_ins) and len(all_tag_ins) and len(word_ins) == len(tag_ins) and len(word_ins) == len(all_tag_ins):
                result.append([word_ins, tag_ins, intent, all_tag_ins])
        fin.close()
        return result
    def load_ner_data(self, train_filename):
        fin = open(train_filename, "r")
        words = []
        tags = []
        word_ins = []
        tag_ins = []
        for line in fin:
            line = line.strip()
            if (len(line) == 0 or line.startswith("-DOCSTART-")):
                if word_ins:
                    words.append(word_ins)
                    tags.append(tag_ins)
                word_ins = []
                tag_ins = []
            else:
                lin = line.split(" ")
                word_txt, tag_txt = lin[0], lin[1]
                word_txt = word_txt.lower()
                if word_txt.isdigit():
                    word_txt = NUM
                if word_txt in self.word_dict:
                    word_ins.append(self.word_dict[word_txt])
                else:
                    word_ins.append(self.word_dict[UNK])
                if tag_txt in self.tag_dict:
                    tag_ins.append(self.tag_dict[tag_txt])
                else:
                    tag_ins.append(self.tag_dict[UNK])
        if word_ins:
            words.append(word_ins)
            tags.append(tag_ins)
        fin.close()
        return zip(words, tags)

def get_chunk_type(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
if __name__ == "__main__":
    config = Config()
    train_file = config.trainfile
    dev_file = config.devfile
    vocabdict = VocabDict(config)
    train = vocabdict.load_flight_data(config.trainfile)
    dev = vocabdict.load_flight_data(config.devfile)
    test = vocabdict.load_flight_data(config.testfile)
    for item in test:
        print item

