import sys,os

conf_dict = {}
fin = open("conf/tree_trans.conf", "r")
for line in fin:
	lin = line.strip().split("\t")
	if len(lin) != 2:
		continue
	new_tag = lin[1].split(":")[-1]
	old_tag = lin[0].split(":")[-1]
	conf_dict[new_tag] = old_tag
fin.close()

for item in conf_dict:
	print item + "\t" + conf_dict[item]

