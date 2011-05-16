#!/usr/bin/env python

# requires installing argparse package.

from argparse import ArgumentParser,FileType
import os,sys,operator,re
import csv, collections
import numpy as np
from mva import mva

# --------------------------------------------------------------------------
# UTILS

def findGr8estKey (dic):
	'''returns the greatest key found in a nested dictionary of two levels
	'''
	keys = dic.keys()
	for subDict in dic.values():
		keys.extend(subDict.keys())
	return max (keys)

def shift (array, offset):
    '''shifts array array offset steps to the right, introducing zeros from the left
    '''
    ret = np.zeros_like(array)
    ret[offset:] = array[0:-offset]
    return ret

def dictToArray (dict):
    size = max(dict.iteritems())[0] + 1
    arr = np.zeros (size)
    for k,v in dict.iteritems():
        arr[k] = v
    return arr

def dictToMatrix (dict):
    size = findGr8estKey (dict) + 1
    mtx = np.zeros ((size,size))
    for row in dict.keys():
		for col in dict[row].keys():
		    mtx[row,col] = dict[row][col]
    return mtx

def normalizeRowWise (mtx):
    '''normalizes the matrix mtx row-wise'''
    size = mtx.shape[0]
    s = np.maximum(np.sum (mtx, axis=1), np.ones((size)))
    s = s.repeat(size).reshape(size,size)
    return np.divide(mtx,s)

def sumMatrices (mtcs):
    '''Sums together a list of matrices. The size of the return matrix will be equal to
    the biggest matrix in the list. (they are assumed to have the same size in all dimensions)
    '''
    # find biggest matrix
    maxM = max(mtcs, key=lambda x: x.shape[0])
    size = maxM.shape[0]
    mtxG = np.zeros(shape=(size, size))
    for m in mtcs:
        n = m.shape[0]
        mtxG[:n,:n] += m
    return mtxG

# end UTILS
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# PRUNING matrices

def pruneP (epsilon):
    '''Used as a filter function, this would filter out all (row,column) pairs not
    containing any value greater than epsilon
    '''
    return lambda(rc): np.where( rc > epsilon)[0].size > 0

def pruneFilter (mtx, predicate):
    '''A boolean vector indicating which rows and cols to keep
    '''
    keeprow = map(predicate, mtx)
    keepcol = map(predicate, mtx.T)
    return map(max, keepcol, keeprow)

def prune (mtx, filter):
    '''Prunes a matrix in all dimensions, based on the boolean vector filter
    '''
    for i in np.arange(mtx.ndim):
        mtx = np.compress(filter, mtx, axis=i)
    return mtx

# end PRUNING
# --------------------------------------------------------------------------



# input is dictionary with tID as key, sequence of lock accesses as value
def countMtxFromSeq(lockSeq):
    '''Computes a nested dictionary with lock transition counts,
	or equivalently, a transition matrix with counts.
    '''
	# use dictionary instead of matrix, since we don't know the total
	# number of locks involved
    lockD = collections.defaultdict(dict)
    last = lockSeq[0][0]
    for lID in lockSeq[1:]:
        if lID[0] in lockD[last]:
            lockD[last][lID[0]] += 1 
        else:
            lockD[last][lID[0]] = 1
        last = lID[0]
    return dictToMatrix(lockD)


# FIX NOW !
def timedTransitions (lockSeq, relLockSeq):
    """Calculates average transition time between locks
    """
    # transition time is:
    # (lockID_0, trylock_0), find following (lockID_0, relLock_0)
    # subtract relLock_0 from trylock_1, as in (lockID_0, trylock_1)
    timeD = collections.defaultdict(dict)
    for i, lID in enumerate(lockSeq[1:]):
        if lID[0] in timeD[relLockSeq[i][0]]:
            timeD[relLockSeq[i][0]][lID[0]] += lID[1] - relLockSeq[i][1]
        else:
            timeD[relLockSeq[i][0]][lID[0]] =  lID[1] - relLockSeq[i][1]
    return timeD


# NOT IN USE, has to be fixed before using
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
    '''parse avg service times from a file on format "lockID avgTime".
    Returns a vector with position lockID having value avgTime
    '''
    lidDict = {}
    recReader = csv.reader (file, delimiter=' ', skipinitialspace=True)
    for _row in recReader:
        lID = int(_row[0])
        time = float(_row[1])
        lidDict[lID] = time
    return dictToArray (lidDict)
	

def lockDictFromRecords(recFile):
    '''Parses a record on the format "tsc tID lID" and returns a dictionary
    where the key is the thread id, and the value is an ordered list of (lID, tsc) tuples.
    Assumes that the list is sorted in increasing order by timestamp
    '''
    tidDict = {}
    recReader = csv.reader (recFile, delimiter=' ', skipinitialspace=True)
    for _row in recReader:
        row = map (int, filter(None, _row))
        if row[1] in tidDict.keys():
            tidDict[row[1]].append((row[2], row[0]))
        else:
            tidDict[row[1]] = [(row[2], row[0])]
    return tidDict

def sumTimeMtx (acqSeq, relSeq):
    return dictToMatrix (timedTransitions (acqSeq, relSeq))

def avgTimeMtx (acqSeq, relSeq, countM):
    '''Generates a matrix containing the avg interarrival times between locks
    '''
    sumTimeM = sumTimeMtx (acqSeq, relSeq)

    size = countM.shape[0]
    if sumTimeM.shape[0] != size:
        print "WARNING: count matrix not same size as interarrival time matrix."
    # calculate avg transition
    r = np.maximum(countM, np.ones((size,size)))
    return np.divide(sumTimeM, r)



def insertIntermediateQs (rMatrix, tMatrix, tArray):
    '''Inserts intermediate queues, supposing that rMatrix is a routing matrix
    for the locks.
    '''
    # create output matrices (twice as big)
    size = rMatrix.shape[0]
    r = np.zeros(shape=(size*2, size*2))
    t = np.zeros(size*2)

    for i,row in enumerate(rMatrix):
        r[2*i,2*i+1]  = 1 # the queue representing the interarrival time always routs into the lock q
        r[(2*i+1),::2] = row # displace each routing by 1

    for i,col in enumerate(tMatrix.T):
        # first row of tMatrix contains all the interarrival times between lock 1
        # and all the following locks
        t[2*i] = np.average(col, weights=rMatrix.T[i]) # aggregate it. (in some way)
        t[2*i+1] = tArray[i]

    return r, t


def pruneAll (rMtx, tMtx, tVec, epsilon):
    '''will prune the input matrices keeping the columns and rows for which
    predicate returns true (on either the column or the row)
    The predicate should have type numpy.ndarray -> bool (or whatever)
    '''
    f = pruneFilter (rMtx, pruneP (epsilon))

    ids = np.compress(f, np.arange(len (f)))
    
    return prune(rMtx, f), prune(tMtx, f), prune(tVec, f), ids

def analyze (acqDic, relDic, servTimeVec, numT):
    print 'calculating matrices'
    cntMtcs = map (countMtxFromSeq, acqDic.values())

    sumInterArrivalMtcs = map (sumTimeMtx, acqDic.values(), relDic.values())

    cntTotalM = sumMatrices (cntMtcs)
    sumInterArrivalTotalM = sumMatrices (sumInterArrivalMtcs)

    # sanity check
    if sumInterArrivalTotalM.shape[0] != cntTotalM.shape[0]:
        print "WARNING: count matrix not same size as interarrival time matrix."

    # FIX, temporary solution
    if cntTotalM.shape[0] != servTimeVec.shape[0]:
        print "WARNING: service time vector not the same size as count matrices. Correcting (not good solution)."
        tVec = np.ones(cntTotalM.shape[0])
        tVec[:servTimeVec.shape[0]] = servTimeVec
    else:
        tVec = servTimeVec

    # calculate avg transition time
    r = np.maximum(cntTotalM, np.ones_like (cntTotalM))
    avgInterArrivalTotalM = np.divide (sumInterArrivalTotalM, r)

    # prune locks not used very much
    cntP, avgInterArrivalP, tVecP, idMap = pruneAll (cntTotalM, avgInterArrivalTotalM, tVec, 1000)

    # normalize (row-wise) the pruned transition count matrix to get the routing matrix
    routP = normalizeRowWise (cntP)

    # insert intermediate infinite server qs to represent the interarrival times
    # service time is calculated as the weighted average of the incoming traffic
    newRout, servTimes = insertIntermediateQs (routP, avgInterArrivalP, tVecP)

    # just to get an idea of which lock is used a lot
    totAccessesEachLock = np.sum (cntP, axis=0)

    # mva it
    estimate = mva (newRout, 1/servTimes, numT)

    print "Waiting times: "
    print estimate[1::2]
    print "Pure service times?: "
    print tVecP
    print "Increased contention?: "
    print estimate[1::2]/tVecP
    print "# of lock accesses:"
    print totAccessesEachLock
    print "row -> lockID map: "
    print [(i, row) for i, row in enumerate (idMap)]
    return 
	
def main():
    global options
    parser = ArgumentParser()
    parser.add_argument("-v", dest="verbose", action="store_true", default=False,
                        help="print more information.")
    parser.add_argument("-d", "--debug",
						action="store_true", dest="debug", default=False,
						help="print debug information")
    parser.add_argument("-n", type=int, dest="numCores", nargs='?', help="Number of customers in queueing network.")
    parser.add_argument("lockacq", help="File containing thread IDs, lock IDs and acquire timestamps, sorted by time.")
    parser.add_argument("lockrel", help="File containing thread IDs, lock IDs and release timestamps, sorted by time.")
    parser.add_argument("servfile", help="File containing lock IDs and average service time, sorted by lock ID.")
    options = parser.parse_args()

    if options.debug:
        print >> sys.stderr, "options:", options

    acqfile = open(options.lockacq)
    acqDic = lockDictFromRecords(acqfile)

    relfile = open(options.lockrel)
    relDic = lockDictFromRecords(relfile)

    servfile = open(options.servfile)
    servTimeVec = waitingTimeParse(servfile)
    
    acqfile.close()
    relfile.close()
    servfile.close()

    analyze (acqDic, relDic, servTimeVec, options.numCores)

    sys.exit(0)
    # END MAIN


# uncomment when running from command line

# if __name__ == '__main__':
#    main() 
