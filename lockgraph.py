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
	lockSeq.pop(0)
	for i, lID in enumerate(lockSeq):
		if lID[0] in timeD[relLockSeq[i][0]]:
			timeD[relLockSeq[i][0]][lID[0]] += lID[1] - relLockSeq[i][1]
		else:
			timeD[relLockSeq[i][0]][lID[0]] =  lID[1] - relLockSeq[i][1]
	return timeD

def waitingTime (acqLockSeq, relLockSeq):
	"""Calculates average waiting time (service time + queue time) per lock
	"""
	# waiting time is:
	# (lockID_0, trylock_0), find following (lockID_0, relLock_0)
	# subtract tryLock_0 from relLock_0
	timeD = collections.defaultdict(dict)
	for i, lID in enumerate(lockSeq):
		if lID[0] in timeD[relLockSeq[count][0]]:
			timeD[relLockSeq[i][0]][lID[0]] += relLockSeq[i][1] - lID[1]
		else:
			timeD[relLockSeq[i][0]][lID[0]] =  relLockSeq[i][1] - lID[1]
	return timeD



def waitingTimeParse (file):
	lidDict = {}
	recReader = csv.reader (file, delimiter=' ', skipinitialspace=True)
	for _row in recReader:
		lID = int(_row[0])
		time = float(_row[1])
		lidDict[lID] = time
	return dictToArray (lidDict)

	

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

def shift (array, offset):
	ret = np.zeros_like(array)
	ret[offset:] = array[0:-offset]
	return ret

def dictToArray (dict):
    size = max(dict.iteritems())[0] + 1
    print size
    arr = np.zeros (size)
    for k,v in dict.iteritems():
        print k
        arr[k] = v
    return arr

def dictToMatrix (dict):
    size = findGr8estKey (dict) + 1
    mtx = np.zeros ((size,size))
    for row in dict.keys():
		for col in dict[row].keys():
		    mtx[row,col] = dict[row][col]
    return mtx

def countMtx (rDict):
	return dictToMatrix(rDict)

def avgTimeMtx (tDict, countM):
	size = countM.shape[0]
	sumTimeM = dictToMatrix(tDict)
	# calculate avg transition
	r = np.maximum(countM, np.ones((size,size)))
	return np.divide(sumTimeM, r)
	
def normalizeRowWise (countM):
	size = countM.shape[0]
	s = np.maximum(np.sum (countM, axis=1), np.ones((size)))
	s = s.repeat(size).reshape(size,size)
	return np.divide(countM,s)
	
def insertIntermediateQs (rMatrix, tMatrix, tArray):
	# first row
	rows, cols = rMatrix.shape
	r = np.zeros(shape=(rows*2, rows*2))
	t = np.zeros(rows*2)
	for i,row in enumerate(rMatrix):
		r[2*i,2*i+1]  = 1 # the queue representing the interarrival time always routs into the lock q
		r[(2*i+1),::2] = row # displace each routing by 1

	for i,col in enumerate(tMatrix.T):
		# first row of tMatrix contains all the interarrival times between lock 1
		# and all the following locks
		# aggregate it. (in some way, not necessarily the average)
		t[2*i] = np.average(col)
		t[2*i+1] = tArray[i]
		
	return r, t

def prune (rMatrix, tMatrix, tArray, predicate):
	'''will prune the matrix r based on the predicate,
	which should have type numpy.ndarray -> bool (or whatever)
	'''
	keepcol = []
	keeprow = []

	for row in rMatrix:
		keeprow.append(predicate(row))

	for col in rMatrix:
		keepcol.append(predicate(col))

	all = map(max, keepcol, keeprow)

	ret = np.compress(all, np.compress(all, rMatrix, axis=0), axis=1)
	ret1 = np.compress(all, np.compress(all, tMatrix, axis=0), axis=1)
	ret2 = np.compress(all, tArray)
	
	return ret, ret1, ret2
	

def aggregateOneThread (acqDic, relDic, tArr):
	transD = transitionsFromSequence (acqDic)
	timeD = timedTransitions (acqDic, relDic)

	countM = countMtx(transD)
	avgTimeM = avgTimeMtx(timeD, countM)

	prunedCount, prunedAvgTime, prunedTArr = prune(countM, avgTimeM, tArr, lambda(rc): np.where( rc > 1000)[0].size > 0)
	routing = normalizeRowWise (prunedCount)
	return routing, prunedAvgTime, prunedTArr



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

	   aggregateOneThread (dic[key], relDic[key])
	   

#	   newRout, timingArr = insertIntermediateQs (routingM, interarrivalM)

	   
    acqfile.close()
    relfile.close()
	
    sys.exit(0)


def graphFromMatrix (mtx):
	tikzgraph = r"\begin{tikzpicture}[->,>=stealth',shorten >=1pt,auto,node distance=2.8cm,semithick]"
	end = r"\end{tikzpicture}"
	abc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVW"
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


#if __name__ == '__main__':
#    main() 
