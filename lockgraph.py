#!/usr/bin/env python

# requires installing argparse package.

from argparse import ArgumentParser,FileType
import os,sys,operator,re
import csv, collections
import numpy as np


# input is dictionary with tID as key, sequence of lock accesses as value
def transitionsFromSequence(lockSeq):
	''' computes a nested dictionary with lock transition counts,
	or equivalently, a transition matrix with counts.
	'''
	# use dictionary instead of matrix, since we don't know the total
	# number of locks involved
	lockD = collections.defaultdict(dict)
	last = lockSeq.pop(0)[0]
	for lID in lockSeq:
		if lID[0] in lockD[last]:
			lockD[last][lID[0]] += 1 
		else:
			lockD[last][lID[0]] = 1
		last = lID[0]
	return lockD


def timedTransitions (lockSeq, relLockSeq):
	"""Calculates average transition time for all lock handovers
	"""
	# transition time is:
	# (lockID_0, trylock_0), find following (lockID_0, relLock_0)
	# subtract relLock_0 from trylock_1, as in (lockID_0, trylock_1)
	timeD = collections.defaultdict(dict)
	count = 0
	lockSeq.pop(0)
	for lID in lockSeq:
		if lID[0] in timeD[relLockSeq[count][0]]:
			timeD[relLockSeq[count][0]][lID[0]] += lID[1] - relLockSeq[count][1]
		else:
			timeD[relLockSeq[count][0]][lID[0]] =  lID[1] - relLockSeq[count][1]
		count += 1
	return timeD

def lockDictFromRecords(recFile):
	# a datastructure containing one entry per thread id.
	# for each thread a list of mutex ids and times will be stored
	tidDict = {}
	recReader = csv.reader (recFile, delimiter=' ', skipinitialspace=True)
	for _row in recReader:
		row = map (int, filter(None, _row))
		if row[1] in tidDict:
			tidDict[row[1]].append((row[2], row[0]))
		else:
			tidDict[row[1]] = [(row[2], row[0])]
	return tidDict

def findGr8estKey (dic):
	'''returns the greatest key found in a nested dictionary of two levels
	'''
	keys = dic.keys()
	for subDict in dic.values():
		keys.extend(subDict.keys())
	return max (keys)

def dictToMatrix (dict):
    size = findGr8estKey (dict) + 1
    mtx = np.zeros ((size,size))
    for row in dict.keys():
		for col in dict[row].keys():
			mtx[row,col] = dict[row][col]
    return mtx

def routingAndTimeMatrix (rDict, tDict):
	size = findGr8estKey (rDict) + 1
	routing = dictToMatrix(rDict)
	timing = dictToMatrix(tDict)
	# calculate avg transition
	r = np.maximum(routing, np.ones((size,size)))
	timing = np.divide(timing, r)
	# normalise routing row-wise
	s = np.maximum(np.sum (routing, axis=1), np.ones((size)))
	s = s.repeat(size).reshape(size,size)
	return (np.divide(routing,s), routing, timing)


def graphFromMatrix (mtx):
	tikzgraph = r"\begin{tikzpicture}[->,>=stealth',shorten >=1pt,auto,node distance=2.8cm,semithick]"
	end = r"\end{tikzpicture}"
	abc = "abcdefghijklmnopqrstuvwxyz"
	count = 0
	nodelist = []
	nodeD = {}
	for row in mtx:
		nodeD[count] = abc[count]
		nodelist.append(r"\node[state] (" + nodeD[count] + r") {$ l_" + str(count) + r"$};")
		count += 1
	tikzgraph += ''.join(nodelist)

	edgelist = [r"\path "]
	rc = 0
	for row in mtx:
		cc = 0
		for col in row:
			if (col > 0.0):
				edgelist.append(r"(" + nodeD[rc] + r") edge node {" + "%.4f" % col + r"} (" + nodeD[cc] + r")")
			cc += 1
		rc += 1
	edgelist.append(';')
	tikzgraph += ''.join(edgelist)
	return tikzgraph + end

def main():
    global options
    parser = ArgumentParser()
    parser.add_argument("-v", dest="verbose", action="store_true", default=False,
                        help="print more information.")
    parser.add_argument("-d", "--debug",
						action="store_true", dest="debug", default=False,
						help="print debug information")
    parser.add_argument("lockacq", help="File containing thread IDs, lock IDs and acquire timestamps, sorted by time.")
    parser.add_argument("lockrel", help="File containing thread IDs, lock IDs and release timestamps, sorted by time.")
    options = parser.parse_args()

    if options.debug:
        print >> sys.stderr, "options:", options

    acqfile = open(options.lockacq)
    dic = lockDictFromRecords(acqfile)

    relfile = open(options.lockrel)
    relDic = lockDictFromRecords(relfile)

    for key in dic:
	   transD = transitionsFromSequence (dic[key])
	   timeD = timedTransitions (dic[key], relDic[key])
	   print timeD
	   transProb, transCount, timing = routingAndTimeMatrix (transD, timeD)
	   print transProb
	   print transCount
	   print timing
	   print graphFromMatrix (transProb)
    acqfile.close()
    relfile.close()
	
    sys.exit(0)
       
if __name__ == '__main__':
    main() 
