#using coding=utf8
import sys,os,time,threading,json
sys.path.append('./gen-py')
sys.path.append('./model')
from dnlu.ttypes import DNluRequest
from dnlu.ttypes import DNluResponse
from dnlu.ttypes import DNluTree
from dnlu.ttypes import DNluSlot
import tensorflow as tf
from common.ttypes import IntentType
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
#from flight_slot_filling import lbs_query_slot_filling
from slot_filling import SlotFillingModel
from data_utils import VocabDict, minibatches,get_chunks
from general_utils import Progbar
import logging
log_format = '%(levelname)s: %(asctime)s: %(process)d '
log_format += '[%(funcName)s() %(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(format = log_format, level = logging.DEBUG)
import time

MAX_LEN = 10
UNK = "$UNK$"
NUM = "$NUM$"
TAG_UNK = "O"
NONE = "O"

slot_tag_map = {}
fin = open("conf/tree_trans.conf", "r")
for line in fin:
	if line.find("#") == 0:
		continue
	lin = line.strip().split("\t")
	if len(lin) != 2:
		continue
	new_tag = lin[1]
	#new_tag = lin[1].split(":")[-1]
	old_tag = lin[0].split(":")[-1]
	if new_tag not in slot_tag_map:
		slot_tag_map[new_tag] = []
	slot_tag_map[new_tag].append(old_tag)
fin.close()

tag_alignment_conf = {}
fin = open("conf/slot_alignment", "r")
for line in fin:
	lin = line.strip().split("\t")
	if len(lin) != 2:
		continue
	tag_alignment_conf[lin[0]] = lin[1]
fin.close()

musicname_dict = {}
fin = open("dict/musicname.playcount", "r")
for line in fin:
	lin = line.strip().split("\t")
	try:
		num = int(lin[1])
	except:
		print >>sys.stderr, "playcount\t" + line.strip()
		continue
	if lin[0] not in musicname_dict:
		musicname_dict[lin[0]] = [-1,-1,[]]
	musicname_dict[lin[0]][0] = num
fin.close()
fin = open("dict/entity.confidence.dict", "r")
for line in fin:
	lin = line.strip().split("\t")
	arr = lin[0].split("\x02")
	if arr[1] != "MusicName":
		continue
	musicname = arr[0]
	if musicname not in musicname_dict:
		musicname_dict[musicname] = [-1, -1, []]
	musicname_dict[musicname][1] = float(lin[1])
fin.close()
fin = open("dict/song_singer.conf", "r")
for line in fin:
	lin = line.strip().split("\t")
	if lin[0] not in musicname_dict:
		musicname_dict[lin[0]] = [-1,-1,[]]
	musicname_dict[lin[0]][2] = lin[1:]
fin.close()

class ServerClass:
	def __init__(self, domain):
		logging.info('load new domain:' + domain)
		tf.reset_default_graph()
		self.config = Config(domain)
		self.vocabdict = VocabDict(self.config)
		self.model = SlotFillingModel(self.config, domain)
		self.model.build(self.config.vocab_size, \
				self.config.dim_word, self.config.hidden_size_lstm, \
				self.config.tag_size)
		self.model.restore_session(self.config.dir_model)

class DNluServiceClient:
	def __init__(self):
		#self.lbs_model = ServerClass("lbs")
		self.music_model = ServerClass("music")
		#self.weather_model = ServerClass("weather")
		#self.news_model = ServerClass("news")
	def print_response(self, response, query):
		trees = response.trees
		for tree in trees:
			intent = tree.intention
			slots = []
			for item in tree.slots:
				slots.append(item.slot + "\x01" + item.tag._node.name \
						+ "\x01" + item.tag._node.ori_text)
		logging.info('dnlu_res_log:\t' + query + "\t" + intent \
				+ "\t" + "\t".join(slots))
	def process(self, query_req):
		res_obj = DNluResponse()
		if not isinstance(query_req, DNluRequest):
			logging.info('error req')
			return res_obj
		logging.info("query begin")
		ia_res = query_req.ia_res.intent2result
		lbs_ia_prob = 0
		weather_ia_prob = 0
		news_ia_prob = 0
		music_ia_prob = 0
		if IntentType.LBS in ia_res:
			lbs_ia_prob = ia_res[IntentType.LBS].intent_prob
		if IntentType.WEATHER in ia_res:
			weather_ia_prob = ia_res[IntentType.WEATHER].intent_prob
		if IntentType.NEWS in ia_res:
			news_ia_prob = ia_res[IntentType.NEWS].intent_prob
		if IntentType.MUSIC in ia_res:
			music_ia_prob = ia_res[IntentType.MUSIC].intent_prob
		"""
		if weather_ia_prob > 0:
			dnlu_res_obj = self.weather_query_slot_filling(query_req)
			if not isinstance(dnlu_res_obj, DNluResponse):
				logging.info("format flight obj error")
				return res_obj
			logging.info("query done")
			return dnlu_res_obj
		if news_ia_prob > 0:
			dnlu_res_obj = self.news_query_slot_filling(query_req)
			if not isinstance(dnlu_res_obj, DNluResponse):
				logging.info("format flight obj error")
				return res_obj
			logging.info("query done")
			return dnlu_res_obj
		"""
		if 0:
		#if lbs_ia_prob > 0:
			logging.info("lbs query")
                        time_start = time.time()
			dnlu_res_obj = self.lbs_query_slot_filling(query_req)
                        time_stop = time.time()
                        logging.info('lbs query slot filling latency:[%s seconds]' % (time_stop - time_start))
			if not isinstance(dnlu_res_obj, DNluResponse):
				logging.info("dnlu res format error")
				return res_obj
			logging.info("query done")
			return dnlu_res_obj
		if music_ia_prob > 0:
			logging.info("music query")
                        time_start = time.time()
			dnlu_res_obj = self.music_query_slot_filling(query_req)
			time_stop = time.time()
			logging.info('music query slot filling latency:[%s seconds]' % (time_stop - time_start))
			if not isinstance(dnlu_res_obj, DNluResponse):
				logging.info("music dnlu res obj error")
				return res_obj
			logging.info("query done")
			return dnlu_res_obj
		return res_obj
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

	def lbs_query_slot_filling(self, req):
		ret = DNluResponse()
		dnlu_res_obj = DNluResponse()
		dnlu_res_obj.trees = []
		ia_res = req.ia_res.intent2result
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
			if tag_name in tag_alignment_conf:
				tag_name = tag_alignment_conf[tag_name]
			tag_text = item._node.ori_text
			logging.info("tag_alignment:" + tag_text + "\x01" + tag_name)
			tag_alignment[tag_text + "\x01" + tag_name] = item
			tag_res.append(tag_text+ "\x01" + tag_name)
			if tag_name not in self.lbs_model.config.vocab_inputtags:
				continue
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
		word_ins, all_tag_ins = self.format_input_once(new_seg_res,\
				input_tag_list, "lbs")
		logging.info(str([word_ins, all_tag_ins]))
		#labels_pred, sequence_lengths, pred_intents, score = \
		#		self.predict_batch(word_ins, all_tag_ins)
		#for word_input, tag_input, lab_pred, intent_pred in \
		#		zip(word_ins, all_tag_ins, labels_pred, )
		#lab_list = self.config.idx2tag[a]
		labels_pred, sequence_lengths, pred_intents, score =\
				self.lbs_model.model.predict_once(word_ins, all_tag_ins)
		logging.info("result:")
		logging.info([labels_pred, pred_intents, score])
		intention = self.lbs_model.config.idx2intent[pred_intents[0]]
		prob = score[0][pred_intents[0]]
		logging.info("intent:" + intention + "\tscore:" + str(prob))
		if intention == "O" or len(new_seg_res) != len(labels_pred[0]):
			logging.info('dnlu_res_log:\t' + query + "\tO")
			return dnlu_res_obj
		tree_ins = DNluTree()
		tree_ins.intention = intention
		tree_ins.prob = prob
		tree_ins.slots = []
		slot_list = []
		for idx,label_res in enumerate(labels_pred[0]):
			slot_ins = DNluSlot()
			slot_tag_name = self.lbs_model.config.idx2tag[label_res]
			slot_list.append(slot_tag_name)
			slot_ins.slot = slot_tag_name#new_seg_res[idx]
			key = intention + ":" + slot_tag_name
			if key not in slot_tag_map:
				if slot_tag_name != "O":
					logging.info("miss tag:\t" + slot_tag_name+"\t" + key)
				continue
			alignment_tag_list = slot_tag_map[key]
			for alignment_tag_name in alignment_tag_list:
				key = new_seg_res[idx] + "\x01" + alignment_tag_name
				if key in tag_alignment:
					slot_ins.tag = tag_alignment[key]
					tree_ins.slots.append(slot_ins)
					break
			#alignment_tag_name = slot_tag_map[key]
			#key = new_seg_res[idx] + "\x01" + alignment_tag_name
			#if key in tag_alignment:
			#	slot_ins.tag = tag_alignment[key]
			#	tree_ins.slots.append(slot_ins)
			#else:
			#	logging.info("miss slot:\t" + key)
		dnlu_res_obj.trees.append(tree_ins)
		logging.info(dnlu_res_obj.trees[0].intention)
		if not isinstance(dnlu_res_obj, DNluResponse):
			logging.info("ret format error")
		else:
			logging.info("ret format right")
		self.print_response(dnlu_res_obj, query)
		return dnlu_res_obj

	def music_query_slot_filling(self, req):
		ret = DNluResponse()
		dnlu_res_obj = DNluResponse()
		dnlu_res_obj.trees = []
		ia_res = req.ia_res.intent2result
		seg_res = req.la_res.tokens
		seg_list = []
		tag_alignment = {}
		tag_res = []
		for item in seg_res:
			seg_list.append(item.term)
		logging.info("begin_request:" + "\t".join(seg_list))
		tag_dict = {}
		person_dict = {}
		for item in req.tag_res:
			tag_name = item._node.name
			if tag_name in tag_alignment_conf:
				tag_name = tag_alignment_conf[tag_name]
			tag_text = item._node.ori_text
			logging.info("tag_alignment:" + tag_text + "\x01" + tag_name)
			tag_alignment[tag_text + "\x01" + tag_name] = item
			tag_res.append(tag_text+ "\x01" + tag_name)
			if tag_name not in self.music_model.config.vocab_inputtags:
				continue
			if tag_text not in tag_dict:
				tag_dict[tag_text] = []
			tag_dict[tag_text].append(tag_name)
			if tag_name == "Person":
				person_dict[tag_text] = True
		####音乐相关需要对musicname进行特殊处理
		for tag_text in tag_dict:
			if "MusicName" not in tag_dict[tag_text]:
				continue
			playcount, entity_confidence, singer_list = musicname_dict[tag_text]
			if playcount >= 10000000:
				tag_dict[tag_text].append("MusicName_HOT")
			if entity_confidence < 0.5:
				tag_dict[tag_text].append("MusicName_BAD")
			for person in person_dict:
				if person in singer_list:
					tag_dict[tag_text].append("MusicName_Singer")
					break
		query = "".join(seg_list)
		new_seg_res = self.max_positive_matching_segment(\
				seg_list, tag_dict)
		logging.info(query + "\t" + "\t".join(new_seg_res))
		input_tag_list = ["O"] * len(new_seg_res)
		for i in range(len(new_seg_res)):
			if new_seg_res[i] in tag_dict:
				itemstr = "\x01".join(tag_dict[new_seg_res[i]])
				input_tag_list[i] = itemstr
		word_ins, all_tag_ins = self.format_input_once(new_seg_res,\
				input_tag_list, "music")
		logging.info("input tag:\t" + "||".join(input_tag_list))
		logging.info(str([word_ins, all_tag_ins]))
		#labels_pred, sequence_lengths, pred_intents, score = \
		#		self.predict_batch(word_ins, all_tag_ins)
		#for word_input, tag_input, lab_pred, intent_pred in \
		#		zip(word_ins, all_tag_ins, labels_pred, )
		#lab_list = self.config.idx2tag[a]
		labels_pred, sequence_lengths, pred_intents, score =\
				self.music_model.model.predict_once(word_ins, all_tag_ins)
		logging.info("result:")
		logging.info([labels_pred, pred_intents, score])
		intention = self.music_model.config.idx2intent[pred_intents[0]]
		prob = score[0][pred_intents[0]]
		logging.info("intent:" + intention + "\tscore:" + str(prob))
		if intention == "O" or len(new_seg_res) != len(labels_pred[0]):
			logging.info('dnlu_res_log:\t' + query + "\tO\t" \
					+ str(len(new_seg_res)) + "\t" + str(len(labels_pred[0])))
			return dnlu_res_obj
		tree_ins = DNluTree()
		tree_ins.intention = intention
		tree_ins.prob = prob
		tree_ins.slots = []
		slot_list = []
		for idx,label_res in enumerate(labels_pred[0]):
			slot_ins = DNluSlot()
			slot_tag_name = self.music_model.config.idx2tag[label_res]
			slot_list.append(slot_tag_name)
			slot_ins.slot = slot_tag_name#new_seg_res[idx]
			key = intention + ":" + slot_tag_name
			if key not in slot_tag_map:
				if slot_tag_name != "O":
					logging.info("miss tag:\t" + slot_tag_name+"\t" + key)
				continue
			alignment_tag_list = slot_tag_map[key]
			for alignment_tag_name in alignment_tag_list:
				key = new_seg_res[idx] + "\x01" + alignment_tag_name
				if key in tag_alignment:
					slot_ins.tag = tag_alignment[key]
					tree_ins.slots.append(slot_ins)
					break
			#alignment_tag_name = slot_tag_map[key]
			#key = new_seg_res[idx] + "\x01" + alignment_tag_name
			#if key in tag_alignment:
			#	slot_ins.tag = tag_alignment[key]
			#	tree_ins.slots.append(slot_ins)
			#else:
			#	logging.info("miss slot:\t" + key)
		slot_output_str = "||".join(slot_list)
		if len(tree_ins.slots) == 0 or slot_output_str == "Slot_Singer":
			tree_ins.intention = "O"
		logging.info("slot output:\t" + slot_output_str)
		dnlu_res_obj.trees.append(tree_ins)
		logging.info(dnlu_res_obj.trees[0].intention)
		if not isinstance(dnlu_res_obj, DNluResponse):
			logging.info("ret format error")
		else:
			logging.info("ret format right")
		self.print_response(dnlu_res_obj, query)
		return dnlu_res_obj
	
	def format_input_once(self, seg_res, input_tag_list, domain):
		word_ins = []
		all_tag_ins = []
		if domain == "lbs":
			domain_model = self.lbs_model
		elif domain == "weather":
			domain_model = self.weather_model
		elif domain == "news":
			domain_model = self.news_model
		elif domain == "music":
			domain_model = self.music_model
		else:
			return word_ins, all_tag_ins
		for idx,item in enumerate(seg_res):
			item = item.lower()
			if input_tag_list[idx].find("TIME") != -1:
				item = "$TIME$"
			elif item.isdigit():
				item = NUM
			if item in domain_model.config.vocab_words:
				word_ins.append(domain_model.config.vocab_words[item])
			else:
				word_ins.append(domain_model.config.vocab_words[UNK])
		for item in input_tag_list:
			item_vec = [0] * domain_model.config.inputtag_size
			if item != "O":
				for subtag in item.split("\x01"):
					if subtag in domain_model.config.vocab_inputtags:
						item_vec[domain_model.config.vocab_inputtags[subtag]] = 1
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
