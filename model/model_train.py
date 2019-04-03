from data_utils import VocabDict
from slot_filling import SlotFillingModel
from config import Config
import sys,os

def train(domain):
    config = Config(domain)
    train_file = config.trainfile
    dev_file = config.devfile
    vocabdict = VocabDict(config)
    train = vocabdict.load_flight_data(config.trainfile)
    dev = vocabdict.load_flight_data(config.devfile)
    model = SlotFillingModel(config, sys.argv[1])
    model.build(config.vocab_size, config.dim_word, config.hidden_size_lstm,\
            config.tag_size)
    model.train(train, dev)

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print >> sys.stderr, "usage print model/train.py {domain}"
		sys.exit(-1)
	train(sys.argv[1])

