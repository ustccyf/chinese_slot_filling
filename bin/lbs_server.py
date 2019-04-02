#using coding=utf8
import sys,os,time,threading,json
sys.path.append('./gen-py')
sys.path.append('./model')
from dnlu.ttypes import DNluRequest
from dnlu.ttypes import DNluResponse
from dnlu.ttypes import DNluTree 
from dnlu.ttypes import DNluSlot 
from nlu.utility.ttypes import TagNode 
from nlu.utility.ttypes import FigType 
from dnlu.DNluService import *
from config import Config
from thrift import Thrift
from thrift.Thrift import TType, TMessageType, TException
from thrift.Thrift import TProcessor
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
#from flight_slot_filling import flight_sf
from slot_filling import SlotFillingModel
from data_utils import VocabDict, minibatches,get_chunks
from general_utils import Progbar
import logging
log_format = '%(levelname)s: %(asctime)s: %(process)d '
log_format += '[%(funcName)s() %(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(format = log_format, level = logging.DEBUG)

MAX_LEN = 6
UNK = "$UNK$"
NUM = "$NUM$"
TAG_UNK = "O"
NONE = "O"

slot_tag_map = {}
fin = open("conf/tree_trans.conf", "r")
for line in fin:
	lin = line.strip().split("\t")
	if len(lin) != 2:
		continue
	new_tag = lin[1].split(":")[-1]
	old_tag = lin[0].split(":")[-1]
	slot_tag_map[new_tag] = old_tag
fin.close()

class DNluServiceClient:
	def __init__(self):
		self.config = Config()
		self.vocabdict = VocabDict(self.config)
		self.model = SlotFillingModel(self.config)
		self.model.build(self.config.vocab_size, self.config.dim_word, self.config.hidden_size_lstm, self.config.tag_size)
		self.model.restore_session(self.config.dir_model)
	
	def process(self, query_req):
		res_obj = DNluResponse()
		if not isinstance(query_req, DNluRequest):
			logging.info('error req')
			return res_obj
		logging.info("query begin")
		response_obj = self.query_sf(query_req)
		if not isinstance(response_obj, DNluResponse):
			logging.info("format class obj error")
			return res_obj
		logging.info("query done")
		return response_obj
	
	def max_positive_matching_segment(self, word_list, tag_dict):
		res_list = []
		break_index = -1
		for i in range(len(word_list)):
			if i <= break_index:
				continue
			flag = 0
			for j in range(MAX_LEN):
				now_index = i + MAX_LEN - j
				if now_index >= word_list:
					continue
				now_word = "".join(word_list[i:now_index])
				if now_word in tag_dict: 
					break_index = now_index - 1
					res_list.append(now_word)
					flag = 1
					break
			if flag == 0:
				res_list.append(word_list[i])
		return res_list

	def query_sf(self, req):
		ret = DNluResponse()
		seg_res = req.la_res.tokens
		seg_list = []
		tag_alignment = {}
		tag_res = []
		for item in seg_res:
			seg_list.append(item.term)
		logging.info("begin_request:" + "\t".join(seg_list))
		tag_dict = {}
		for item in req.tag_res:
			tag_name = item._node.name
			tag_text = item._node.ori_text
			tag_alignment[tag_text + "\x01" + tag_name] = item
			tag_res.append(tag_text+ "\x01" + tag_name)
			if tag_text not in tag_dict:
				tag_dict[tag_text] = []
			tag_dict[tag_text].append(tag_name)
		query = "".join(seg_list)
		new_seg_res = self.max_positive_matching_segment(\
				seg_list, tag_dict)
		logging.info(query + "\t" + "\t".join(new_seg_res))
		input_tag_list = ["O"] * len(new_seg_res)
		for i in range(len(new_seg_res)):
			if new_seg_res[i] in tag_dict:
				itemstr = "\x01".join(tag_dict[new_seg_res[i]])
				input_tag_list[i] = itemstr
		word_ins, all_tag_ins = self.format_input_once(new_seg_res, input_tag_list)
		logging.info(str([word_ins, all_tag_ins]))
		#labels_pred, sequence_lengths, pred_intents, score = \
		#		self.predict_batch(word_ins, all_tag_ins)
		#for word_input, tag_input, lab_pred, intent_pred in \
		#		zip(word_ins, all_tag_ins, labels_pred, )
		#lab_list = self.config.idx2tag[a]
		labels_pred, sequence_lengths, pred_intents, score =\
				self.model.predict_once(word_ins, all_tag_ins)
		logging.info("result:")
		logging.info([labels_pred, pred_intents, score])
		intention = self.config.idx2intent[pred_intents[0]]
		prob = score[0][pred_intents[0]]
		logging.info("intent:" + intention + "\tscore:" + str(prob))
		response_obj = DNluResponse()
		response_obj.trees = []
		if intention == "O":
			return response_obj
		tree_ins = DNluTree()
		tree_ins.intention = intention
		tree_ins.prob = prob
		tree_ins.slots = []
		if len(new_seg_res) != len(labels_pred[0]):
			return ret
		for idx,label_res in enumerate(labels_pred[0]):
			slot_ins = DNluSlot()
			slot_tag_name = self.config.idx2tag[label_res]
			slot_ins.slot = slot_tag_name#new_seg_res[idx]
			if slot_tag_name not in slot_tag_map:
				continue
			alignment_tag_name = slot_tag_map[slot_tag_name]
			key = new_seg_res[idx] + "\x01" + alignment_tag_name
			if key in tag_alignment:
				slot_ins.tag = tag_alignment[key]
				tree_ins.slots.append(slot_ins)
		response_obj.trees.append(tree_ins)
		logging.info(response_obj.trees[0].intention)
		if not isinstance(response_obj, DNluResponse):
			logging.info("ret format error")
		else:
			logging.info("ret format right")
		return response_obj

	def format_input_once(self, seg_res, input_tag_list):
		word_ins = []
		all_tag_ins = []
		for idx,item in enumerate(seg_res):
			item = item.lower()
			if input_tag_list[idx].find("TIME") != -1:
				item = "$TIME$"
			elif item.isdigit():
				item = NUM
			if item in self.config.vocab_words:
				word_ins.append(self.config.vocab_words[item])
			else:
				word_ins.append(self.config.vocab_words[UNK])
		for item in input_tag_list:
			item_vec = [0] * self.config.inputtag_size
			if item != "O":
				for subtag in item.split("\x01"):
					if subtag in self.config.vocab_inputtags:
						item_vec[self.config.vocab_inputtags[subtag]] = 1
			all_tag_ins.append(item_vec)
		return word_ins, all_tag_ins



def run(server_port):
	handler = DNluServiceClient()
	processor = Processor(handler)
	transport = TSocket.TServerSocket(port = server_port)
	tfactory = TTransport.TBufferedTransportFactory()
	pfactory = TBinaryProtocol.TBinaryProtocolFactory()
	server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
	server.setNumThreads(10)
	logging.info('start thrift serve in python')
	server.serve()
	logging.info('done!')

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print >>sys.stderr, "usage python bin/server.py port"
		exit(-1)
	server_port = sys.argv[1]
	run(server_port)
