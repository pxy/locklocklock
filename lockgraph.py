"""SLAP-
$ Time-stamp: <2011-08-01 14:25:58 jonatanlinden>

README:
A collection of tools to do a queueing network analysis on sequences
of timestamps representing threads lock accesses.

datafiles should have row format 'timestamp threadID lockID', and
should be sorted by the timestamp.
"""

from collections import defaultdict
import os,re,csv,subprocess
import numpy as np
from mva import mva
import histo

# CONSTANTS

ADDR2LINEPATH = "addr2line-x86_64-elf"


# we capsulate the data of one specific measurement in an object




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
    for row in dict:
		for col in dict[row]:
		    mtx[row,col] = dict[row][col]
    return mtx

def normalizeRowWise (mtx):
    '''normalizes the matrix mtx row-wise'''
    size = mtx.shape[0]
	# sum of each row in s (minimum 1)
    s = np.maximum(np.sum (mtx, axis=1), np.ones((size)))
	# divide every row of mtx by s
    return np.divide(mtx.T,s).T

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

def revertListDict (dict):
    '''dict : a dictionary with lists as values
    PRE : each value in each of the lists are unique
    '''
    retD = {}
    for k,v in dict.iteritems():
        for i in v:
            retD[i] = k
    return retD

def collapseLevel (listofdict):
	'''INPUT: list of dictionaries
	OUTPUT: dictionary with keys from dictionaries
	'''
	retD = defaultdict(list)
	for dic in listofdict:
		for k1 in dic: 
			retD[k1].extend(dic[k1])
	return retD

def listOfArrToMtx (list):
	'''list: a list of variable length array-like elements
	POST: a matrix whose rows consists of the elements in list
	'''
	mtx = np.zeros ((len(list), len(max (list, key=len))))

	for i, row in enumerate (list):
		mtx[i,0:len(row)] = row
	return mtx

def imax(l):
    return l.index(max(l))


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
    '''POST: A boolean vector indicating which rows and cols to keep
    '''
    keeprow = map(predicate, mtx)
    keepcol = map(predicate, mtx.T)
    return map(max, keepcol, keeprow)

def filterByLockName (lockNames, filterName):
    '''lockNames should be a list of strings where lock with id 0 corresponds to
    the string at pos 0 in the list
    '''
    regex = re.compile(filterName)
    res = map(lambda x: bool(regex.search(x)), lockNames)
    print res
    return

def prune (mtx, filter):
    '''Prunes a matrix in all dimensions, based on the boolean vector filter
    '''
    for i in np.arange(mtx.ndim):
        mtx = np.compress(filter, mtx, axis=i)
    return mtx

# end PRUNING
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# PARSING CVS 

def creationParse(file):
    lockCreateList = []
    recReader = csv.reader (file, delimiter=' ', skipinitialspace=True)
    for _row in recReader:
        lockCreateList.append(_row[1])
    return lockCreateList


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
	'''Parses a record on the format "tsc tID lID" and returns a
	dictionary where the key is the thread id, and the value is an
	ordered list of (lID, tsc) tuples. Assumes that the list is sorted
	in increasing order by timestamp
	'''
	tidDict = defaultdict(list)
	recReader = csv.reader (recFile, delimiter=' ', skipinitialspace=True)
	for _row in recReader:
		row = map (int, filter(None, _row))
		tidDict[row[1]].append((row[2], row[0]))
	return tidDict

def parseInstrList (instrFile):
    '''Input file should be on the format "lID randomcrap [hexaddr]"
    '''
    regex = re.compile("\[(?P<hexaddr>\w+)\]")
    res = []
    recReader = csv.reader (instrFile, delimiter=' ', skipinitialspace=True)
    for _row in recReader:
        res.append(regex.search(_row[2]).group('hexaddr'))
    return res

#--------------------------------------------------------------------------
# ENTRY POINT for parsing
#--------------------------------------------------------------------------

def parseDicTuple (prefix):
	'''Parses four related files, based on a file path prefix and the number
	of threads the files represent.
	Assumes the data files follow the naming convention prefix_{try,acq,rel,addr}.dat
	prefix should be the path, including the appname.
	'''
	tryfile = open(prefix + "_try.dat")
	tryDic = lockDictFromRecords(tryfile)

	acqfile = open(prefix + "_acq.dat")
	acqDic = lockDictFromRecords(acqfile)

	relfile = open(prefix  + "_rel.dat")
	relDic = lockDictFromRecords(relfile)

	creationFile = open(prefix + "_addr.dat")
	createVec = creationParse(creationFile)
	#creationFile.seek(0)

	#instrVec = parseInstrList(creationFile)

	tryfile.close()
	acqfile.close()
	relfile.close()
	creationFile.close()

	return (tryDic, acqDic, relDic, createVec)


# end PARSING
# --------------------------------------------------------------------------

def lockD (acqLockDs, relLockDs):
	waitingTimes = map (waitingTime, acqLockDs.values(), relLockDs.values())
	lockD = collapseLevel (waitingTimes)
	return lockD



def countMtxFromSeq(lockSeq):
	'''Computes a nested dictionary with lock transition counts,
	or equivalently, a transition matrix with counts from the lock sequence of
	a specific thread.
	'''
	# use dictionary instead of matrix, since we don't know the total
	# number of locks involved
	lockD = defaultdict(lambda : defaultdict(int))
	last = lockSeq[0][0]
	for lID in lockSeq[1:]:
		lockD[last][lID[0]] += 1
		last = lID[0]
	return dictToMatrix(lockD)

# FIX NOW !
def timedTransitions (lockSeq, relLockSeq):
	"""Calculates average transition time between locks
	"""
	# transition time is:
	# (lockID_0, trylock_0), find following (lockID_0, relLock_0)
	# subtract relLock_0 from trylock_1, as in (lockID_0, trylock_1)
	timeD = defaultdict(lambda : defaultdict(int))
	for i, lID in enumerate(lockSeq[1:]):
		timeD[relLockSeq[i][0]][lID[0]] += lID[1] - relLockSeq[i][1]
	return timeD


def waitingTime (tryLockSeq, relLockSeq):
	'''POST: Dictionary containing all individual waiting times for each lock
	'''
    # waiting time is:
    # (lockID_0, trylock_0), find following (lockID_0, relLock_0) in rel
    # subtract tryLock_0 from relLock_0
	timeD = defaultdict(list)
	for i, tryL in enumerate(tryLockSeq):
		rel = relLockSeq[i]
		if tryL[0] != rel[0]:
			print "ERROR: lock sequences not synced"
		timeD[rel[0]].append(rel[1] - tryL[1])
	return timeD

def sumWaitingTime (trySeq, relSeq):
	sumWait = 0
	for i, tryL in enumerate(trySeq):
		rel = rellSeq[i]
		if tryL[0] != rel[0]:
			print "ERROR: lock sequences not synced"
		sumWait += rel[1] - tryL[1]
	return sumWait



    
def avgWaitTime (tryLockSeq, relLockSeq, perc):
	"""Calculates average waiting time (service time + queue time) per lock
	INPUT: tryLockSeq, relLockSeq : a tuple list of the form (lockID, timestamp)
	"""
	# create a dictionary indexed by lockIDs
	timeD = waitingTime (tryLockSeq, relLockSeq)
	size = max (timeD.keys()) + 1
	arr = np.zeros(size)
	# the number of accesses to each lock will be defined by the length of
	# each dictionary value
	countArr = np.zeros(size)
	for k,v in timeD.iteritems():
		# assumes lock ids appear in order without gaps
		countArr[k] = len(v)
		v.sort()
		lo = (100-perc)*len(v)/100
		hi = perc*len(v)/100
		if hi == lo:
			hi = len (v)
			lo = 0
		arr[k] = float(sum(v[lo:hi])) / (hi - lo)
	return (arr, countArr)

def servTime (acqD, relD, perc):
	servTimeList = []
	for k in acqD:
		servTimeList.append(avgWaitTime(acqD[k], relD[k], perc))

	servTimes, counts = zip (*servTimeList)

	servTimesMtx = listOfArrToMtx (servTimes)
	countMtx = listOfArrToMtx (counts)

	# norm of columns
	norms = normalizeRowWise(countMtx.T).T
	
	return np.average (servTimesMtx, axis=0, weights=norms)
	

def sumTimeMtx (trySeq, relSeq):
    return dictToMatrix (timedTransitions (trySeq, relSeq))

def avgTimeMtx (trySeq, relSeq, countM):
    '''Generates a matrix containing the avg interarrival times between locks
    '''
    sumTimeM = sumTimeMtx (trySeq, relSeq)

    size = countM.shape[0]
    if sumTimeM.shape[0] != size:
        print "WARNING: count matrix not same size as interarrival time matrix."
    # calculate avg transition
    r = np.maximum(countM, np.ones((size,size)))
    return np.divide(sumTimeM, r)


def hexToLine (createVec_, appname):
	'''converts each hex address in createVec to a filename and row number
	'''
	process = subprocess.Popen([ADDR2LINEPATH, '-e', appname], shell=False,
							   stdout=subprocess.PIPE,
							   stdin=subprocess.PIPE
							   )
	createVec = []
	for addr in createVec_:
		process.stdin.write(addr + '\n') # write address to addr2line, fetch line
		createVec.append(process.stdout.readline().rpartition('/')[2].rstrip())

	process.terminate()
	return createVec


def lockToServTimeMap (instrVec, servTime, countVec):
	'''POST: a map from instructions to weighted averaged service times
	'''
	lockD1 = defaultdict(int)
	for i,k in enumerate(instrVec):
		lockD1[k] += i

	# servD1 instr -> service time
	servD1 = {}
	for k in lockD1:
		filtered = []
		keys = []
		for x in lockD1[k]:
			if x < len(servTime):
				keys.append(x)
				filtered.append(servTime[x])
		norm_ = []
		for key in keys:
			norm_.append(countVec[key])
		sumn = sum(norm_)
		norm = np.array(norm_)
		norm /= sumn * 1.0
		servD1[k] = sum(np.array(filtered)*norm)
	return servD1


def lockMap (createVec1_, createVec2_, servTimeVec2, countVec2):
    '''Creates a surjective map from vector 1 to vector 2
    createVec1_ and createVec2_ should be lists of some text uniquely
    identifying a piece of code
    '''
    # create dict indexed by string (filename + linenumber) from 1 th case
    servD1 = lockToServTimeMap (createVec2_, servTimeVec2, countVec2)

    lockD = defaultdict(list)            
    for i,k in enumerate(createVec1_):
		lockD[k].append(i)

    # lockD instr -> lockID list
    mapp = revertListDict (lockD)

    # TODO: if one lock (row of code) does not appear in lockD1, what to do?
	# FIX: correct missing indices manually
    #retD = {}
    #for k in mapp.keys():
    #    retD[k] = servD1[mapp[k]]
	#servTimeMapp = dictToArray (retD)
    return servD1, mapp


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


def add_overhead (vector):
	'''vector: array-like, times in cycles
	'''
	return vector + 4000/0.4 # 4 microsecs of overhead


def routing (tryDic, acqDic, relDic, namesVec, tIDs):
	cntMtcs = map (lambda x: countMtxFromSeq(acqDic[x]), tIDs)
	cntTotalM = sumMatrices (cntMtcs)

	sumInterArrivalMtcs = []
	for i in tIDs:
		sumInterArrivalMtcs.append(sumTimeMtx(tryDic[i], relDic[i]))

	cntTotalM = sumMatrices (cntMtcs)
	#sumInterArrivalTotalM = sumMatrices (sumInterArrivalMtcs)

	# sanity check
	#if sumInterArrivalTotalM.shape[0] != cntTotalM.shape[0]:
	#	print "WARNING: count matrix not same size as interarrival time matrix."

	#servTimeVec_ = servTime (acqDic, relDic, 95)
	#servTimeVec = servTimeVec_

	# calculate avg transition time
	#r = np.maximum(cntTotalM, np.ones_like (cntTotalM))
	#avgInterArrivalTotalM = np.divide (sumInterArrivalTotalM, r)
	#rout = normalizeRowWise (cntTotalM)
	return cntTotalM

def sumPredictedWaitingTime (waitVec, countVec):
	return sum (map (operator.mul, waitVec, countVec))

#--------------------------------------------------------------------------
# entry point of application

def analyze (tryDic, acqDic, relDic, namesVec, numT):
	cntMtcs = map (countMtxFromSeq, acqDic.values())

	sumInterArrivalMtcs = []
	for i in tryDic.keys():
		sumInterArrivalMtcs.append(sumTimeMtx(tryDic[i], relDic[i]))

	cntTotalM = sumMatrices (cntMtcs)
	sumInterArrivalTotalM = sumMatrices (sumInterArrivalMtcs)

	# sanity check
	if sumInterArrivalTotalM.shape[0] != cntTotalM.shape[0]:
		print "WARNING: count matrix not same size as interarrival time matrix."

	servTimeVec_ = servTime (acqDic, relDic, 95)

	servTimeVec = servTimeVec_
	#servTimeVec = add_overhead(servTimeVec_)
	

	# calculate avg transition time
	r = np.maximum(cntTotalM, np.ones_like (cntTotalM))
	avgInterArrivalTotalM = np.divide (sumInterArrivalTotalM, r)

	# prune locks not used very much
	#cntP, avgInterArrivalP, tVecP, idMap = pruneAll (cntTotalM, avgInterArrivalTotalM, servTimeVec, 100)
	# normalize (row-wise) the pruned transition count matrix to get the routing matrix
	#routP = normalizeRowWise (cntP)
	# insert intermediate infinite server qs to represent the interarrival times
	# service time is calculated as the weighted average of the incoming traffic
	#newRout, servTimes = insertIntermediateQs (routP, avgInterArrivalP, tVecP)


	rout = normalizeRowWise (cntTotalM)
	newRout, servTimes = insertIntermediateQs (rout, avgInterArrivalTotalM, servTimeVec)

	# just to get an idea of which lock is used a lot
	#totAccessesEachLock = np.sum (cntP, axis=0)

	cntTot = np.sum (cntTotalM, axis=0)

	# mva it
	estimate = mva (newRout, 1/servTimes, numT)
	estincr  = estimate[1::2]/servTimeVec

	# actual waiting time for numT threads
	actualWait = servTime(tryDic, relDic, 95)

	for i,e in enumerate (estimate[1::2]):
		print '%s : act: %6.0f, est: %6.0f, serv: %6.0f, est.incr: %1.3f, acc: %d' % (namesVec[i], actualWait[i], estimate[1::2][i], servTimeVec[i], estincr[i], cntTot[i])

	return zip (namesVec, actualWait, estimate[1::2], servTimeVec)

