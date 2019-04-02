import sys,os,json
from nlu.utility.ttypes import DNliRequest
from nlu.utility.ttypes import DNluResult
MAX_LEN = 6

def max_positive_matching_segment(word_list, tag_dict):
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

def flight_sf(req, model):
	ret = DNliRequest()
	seg_res = req.la_res.tokens
	seg_list = []
	tag_res = []
	for item in seg_res:
		seg_list.append(item.term)
	tag_dict = {}
	for item in req.tag_res:
		tag_name = item._node.name
		tag_text = item._node.ori_text
		tag_res.append(tag_text+ "\x01" + tag_name)
		if tag_text not in tag_dict:
			tag_dict[tag_text] = []
		tag_dict[tag_text].append(tag_name)
	query = "".join(seg_list)
	new_seg_res = max_positive_matching_segment(\
			seg_list, tag_dict)
	input_tag_list = ["O"] * len(new_seg_res)
	for i in range(len(new_seg_res)):
		if new_seg_res[i] in tag_dict:
			itemstr = "\x01".join(tag_dict[new_seg_res[i]])
			input_tag_list[i] = itemstr
	print query + "\t" + "\x01".join(new_seg_res) + "\t" \
			+ "\t".join(input_tag_list)
	#print query + "\t" + "\x01".join(seg_list) + "\t" + \
	#		"\t".join(tag_res)
	return ret

def format_input_once(seg_res, input_tag_list):


