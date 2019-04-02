import sys,os,time,threading,json
sys.path.append('./gen-py')

from dnlu.ttypes import DNluRequest
from dnlu.ttypes import DNluResponse
from dnlu.DNluService import *

from thrift import Thrift
from thrift.Thrift import TType, TMessageType, TException
from thrift.Thrift import TProcessor
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
#from flight_slot_filling import flight_sf
from thrift.transport.TTransport import TMemoryBuffer
#from thrift.protocol.TBinaryProtocol import TBinaryProtocol, TBinaryProtocolAccelerated
import logging
log_format = '%(levelname)s: %(asctime)s: %(process)d '
log_format += '[%(funcName)s() %(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(format = log_format, level = logging.DEBUG)

def client_flight_sf(data_list, thread_num, server_port):
	try:
		transport = TSocket.TSocket('localhost', server_port)
		transport = TTransport.TBufferedTransport(transport)
		protocol = TBinaryProtocol.TBinaryProtocol(transport)
		client = Client(protocol)
		transport.open()
		for req in data_list:
			res = client.process(req)
			if res == None or not isinstance(res, DNluResponse):
				print "error res"
				continue
			query = ""
			for item in req.la_res.tokens:
				query += item.term
			for tree in res.trees:
				slot_info = []
				for slot in tree.slots:
					slot_info.append(slot.slot + "\x01" + slot.tag._node.name + "\x01" + slot.tag._node.ori_text)
				print query  + "\t" + tree.intention + "\t" + str(tree.prob) + "\t"\
					+ "\t".join(slot_info)
		transport.close()
	except Thrift.TException as e:
		print 'exceptino'
		print e

if __name__ == "__main__":
	count = 0
	file_name = sys.argv[1]
	server_port = int(sys.argv[2])
	data_list = []
	count = 0
	fin = open(file_name, "r")
	for line in fin:
		line = line.strip()
		lin = line.split("\t")
		if len(lin) < 2 or lin[0] != "dl_req_str":
			continue
		line = "\t".join(lin[1:]).replace("cyf_split", "\n")
		memory_buffer = line.rstrip("\n")
		req = DNluRequest()
		tMemory_o = TMemoryBuffer(memory_buffer)
		tBinaryProtocol_o = TBinaryProtocol.TBinaryProtocol(tMemory_o)
		req.read(tBinaryProtocol_o)
		data_list.append(req)
	fin.close()
	thread_count = 1
	thread_list = []
	for i in range(thread_count):
		thread_inst = threading.Thread(target=client_flight_sf, \
				args=(data_list, i, server_port))
		thread_list.append(thread_inst)
	time_start = time.time()
	for thread_inst in thread_list:
		thread_inst.start()
	for thread_inst in thread_list:
		thread_inst.join()
	time_end = time.time()
	all_time = time_end - time_start
	count = len(data_list)
	qps = count * thread_count/float(all_time)
	print >>sys.stderr, "query:" + str(count) + "\tall_time:" + str(all_time)\
			+ "\tqps:" + str(qps)

