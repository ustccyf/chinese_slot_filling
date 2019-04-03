from data_utils import VocabDict
from slot_filling import SlotFillingModel
from config import Config
import sys,os

def main(domain):
    config = Config(domain)
    model = SlotFillingModel(config, domain)
    model.build(config.vocab_size, config.dim_word, config.hidden_size_lstm, config.tag_size)
    model.restore_session(config.dir_model)
    vocabdict = VocabDict(config)
    #train = vocabdict.load_flight_data(config.trainfile)
    #model.evaluate(train, True)
    #dev = vocabdict.load_flight_data(config.devfile)
    #model.evaluate(dev, True)
    test = vocabdict.load_flight_data(config.testfile)
    model.evaluate(test, True)
if __name__ == "__main__":
    main(sys.argv[1])
