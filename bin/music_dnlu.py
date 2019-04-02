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
	lin = line.strip().split("\t")
	if len(lin) != 2:
		continue
	new_tag = lin[1]
	#new_tag = lin[1].split(":")[-1]
	old_tag = lin[0].split(":")[-1]
	slot_tag_map[new_tag] = old_tag
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
		if tag_name not in self.lbs_model.config.vocab_inputtags:
			continue
		if tag_text not in tag_dict:
			tag_dict[tag_text] = []
		tag_dict[tag_text].append(tag_name)
		if tag_name == "Person":
			person_dict[tag_text] = True
	####音乐相关需要对musicname进行特殊处理
	for tag_text in tag_dict:
		if "Musicname" not in tag_dict[tag_text]:
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
	if intention == "O":
		logging.info('dnlu_res_log:\t' + query + "\tO")
		return dnlu_res_obj
	tree_ins = DNluTree()
	tree_ins.intention = intention
	tree_ins.prob = prob
	tree_ins.slots = []
	if len(new_seg_res) != len(labels_pred[0]):
		return ret
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
		alignment_tag_name = slot_tag_map[key]
		key = new_seg_res[idx] + "\x01" + alignment_tag_name
		if key in tag_alignment:
			slot_ins.tag = tag_alignment[key]
			tree_ins.slots.append(slot_ins)
		else:
			logging.info("miss slot:\t" + key)
	slot_output_str = "||".join(slot_list)
	if len(tree_ins.slots) == 0 or slot_output_str == "Slot_Singer":
		dnlu_res_obj.trees[0].intention = "O"
	dnlu_res_obj.trees.append(tree_ins)
	logging.info(dnlu_res_obj.trees[0].intention)
	if not isinstance(dnlu_res_obj, DNluResponse):
		logging.info("ret format error")
	else:
		logging.info("ret format right")
	self.print_response(dnlu_res_obj, query)
	return dnlu_res_obj

