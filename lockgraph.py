#!/usr/bin/env python

# requires installing argparse package.

from argparse import ArgumentParser,FileType
import os,sys,operator,re
import csv
import numpy as np


# input is dictionary with tID as key, sequence of lock accesses as value
def transitionsFromSequence(lockSeq):
	''' computes a routing matrix from a dictionary
	'''
	lockD = {}
	last = lockSeq[0]
	for lID in lockSeq:
		if last in lockD:
			if lID in lockD[last]:
				lockD[last][lID] += 1
			else:
				lockD[last][lID] = 1
		else:
			lockD[last] = {}
			lockD[last][lID] = 1
		last = lID
	return lockD

def lockDictFromRecords(recFile):
	# a datastructure containing one entry per thread id.
	# for each thread a list of mutex ids and times will be stored
	tidDict = {}
	recReader = csv.reader (recFile, delimiter=' ', skipinitialspace=True)
	for row in recReader:
		num = map (int, filter(None, row))
		if num[1] in tidDict:
			tidDict[num[1]].append(num[2])
		else:
			tidDict[num[1]] = [num[2]]
	return tidDict

def findGr8estKey (dic):
	keys = dic.keys()
	for subDict in dic.values():
		keys.extend(subDict.keys())
	return max (keys)

def matrixFromDict (dict):
	print dict
	size = findGr8estKey (dict) + 1
	routing = np.zeros ((size,size))
	for row in dict.keys():
		for col in dict[row].keys():
			routing[row,col] = dict[row][col]
	s = np.sum (routing, axis=1)
	s = np.maximum(s, np.ones((size)))
	s2 = s.repeat(size).reshape(size,size)
	print s2
	print routing
	return np.divide(routing,s2)


def main():
    global options
    parser = ArgumentParser()
    parser.add_argument("-v", dest="verbose", action="store_true", default=False,
                        help="print more information.")
	# parser.add_argument("--name", dest="name", help="name of parameter to change in net-file.")
    # parser.add_argument("--val", dest="val", type=int, help="the new value to set.")
    parser.add_argument("inputfile", help="no help.")
    parser.add_argument("-d", "--debug",
						action="store_true", dest="debug", default=False,
						help="print debug information")

    options = parser.parse_args()

    if options.debug:
        print >> sys.stderr, "options:", options

    netfile  = open(options.inputfile)
    dic = lockDictFromRecords(netfile)

    for key in dic:
	   transD = transitionsFromSequence (dic[key])
	   print matrixFromDict (transD)
	
    sys.exit(0)
       
if __name__ == '__main__':
    main() 
