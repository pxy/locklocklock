#!/usr/bin/env python

# requires installing argparse package.

from argparse import ArgumentParser,FileType
import os,sys,operator,re
import csv, collections
import numpy as np
import subprocess
from mva import mva

# CONSTANTS

ADDR2LINEPATH = "addr2line-x86_64-elf"


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
    tidDict = {}
    recReader = csv.reader (recFile, delimiter=' ', skipinitialspace=True)
    for _row in recReader:
        row = map (int, filter(None, _row))
        if row[1] in tidDict.keys():
            tidDict[row[1]].append((row[2], row[0]))
        else:
            tidDict[row[1]] = [(row[2], row[0])]
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



def parseDicTuple (prefix, ths):
    acqfile = open(prefix + "_" + str(ths) + "th_acq.dat")
    acqDic = lockDictFromRecords(acqfile)

    relfile = open(prefix + "_" + str(ths) + "th_rel.dat")
    relDic = lockDictFromRecords(relfile)

    creationFile = open(prefix + "_lid_instr_" + str(ths) + "th.dat")
    createVec = creationParse(creationFile)

    creationFile.seek(0)
    instrVec = parseInstrList(creationFile)

    acqfile.close()
    relfile.close()
    creationFile.close()
    
    return (acqDic, relDic, createVec, instrVec)



# end PARSING
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


def waitingTime (acqLockSeq, relLockSeq):
    """Calculates average waiting time (service time + queue time) per lock
    acqLockSeq, relLockSeq : a tuple list of the form (lockID, timestamp)
    """
    # waiting time is:
    # (lockID_0, trylock_0), find following (lockID_0, relLock_0) in rel
    # subtract tryLock_0 from relLock_0
    timeD = collections.defaultdict(dict)
    for i, acq in enumerate(acqLockSeq):
        rel = relLockSeq[i]
        if acq[0] != rel[0]:
            print "ERROR: lock sequences not synced"
        if acq[0] in timeD.keys():
            timeD[rel[0]].append(rel[1] - acq[1])
        else:
            timeD[rel[0]] = [rel[1] - acq[1]]
    size = max (timeD.keys()) + 1
    arr = np.zeros(size)
    for k,v in timeD.iteritems():
        arr[k] = float(sum (v))/len(v)
    return arr


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

def hexToLine (createVec_, appname):
    # convert each hex address to line in code for both lists
    createVec = []

    process = subprocess.Popen([ADDR2LINEPATH, '-e', appname], shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    for addr in createVec_:
        process.stdin.write(addr + '\n')
        createVec.append(process.stdout.readline().rpartition('/')[2].rstrip())

    process.terminate()
    
    return createVec


def revertListDict (dict):
    '''dict : a dictionary with lists as values
    PRE : each value in each of the lists are unique
    '''
    retD = {}
    for k,v in dict.iteritems():
        for i in v:
            retD[i] = k
    return retD

def lockToServTimeMap (instrVec, servTime):
    lockD1 = {}
    for i,k in enumerate(instrVec):
        if k in lockD1.keys():
            lockD1[k].append(i)
        else:
            lockD1[k] = [i]

    # lockD1 instr -> lockID list

    # servD1 instr -> service time
    servD1 = {}
    for k in lockD1.keys():
        print k, lockD1[k]
        filtered = []
        for x in lockD1[k]:
            if x < len(servTime):
                filtered.append(servTime[x])
        servD1[k] = float(sum(filtered)) / len(filtered)
    return servD1

def lockMap (createVec1_, createVec2_, servTimeVec2):
    '''Creates a surjective map from vector 1 to vector 2
    createVec1_ and createVec2_ should be lists of some text uniquely
    identifying a piece of code
    '''
    # create dict indexed by string (filename + linenumber) from 1 th case

    servD1 = lockToServTimeMap (createVec2_, servTimeVec2)

    lockD = {}            
    for i,k in enumerate(createVec1_):
        if k in lockD.keys():
            lockD[k].append(i)
        else:
            lockD[k] = [i]

    print servD1

    # lockD instr -> lockID list
    mapp = revertListDict (lockD)

    # create surjective mapping from values (8 th case) to keys (1 th case)

    # TODO: if one lock (row of code) does not appear in lockD1, what to do?
    #retD = {}
    #for k in mapp.keys():
    #    retD[k] = servD1[mapp[k]]
    return servD1, mapp
    #fix missing indices manually


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

def analyze (acqDic, relDic, servTimeVec_, numT, lockNames):
    print 'calculating matrices'
    cntMtcs = map (countMtxFromSeq, acqDic.values())

    sumInterArrivalMtcs = map (sumTimeMtx, acqDic.values(), relDic.values())

    cntTotalM = sumMatrices (cntMtcs)
    sumInterArrivalTotalM = sumMatrices (sumInterArrivalMtcs)

    # sanity check
    if sumInterArrivalTotalM.shape[0] != cntTotalM.shape[0]:
        print "WARNING: count matrix not same size as interarrival time matrix."


    
    # After this point, the mapping between locks in the service time array and the
    # matrices should be one to one

    servTimeVec = servTimeVec_[0:cntTotalM.shape[0]]
        
    
    #if cntTotalM.shape[0] != servTimeVec.shape[0]:
    #    print "ERROR: service time vector not the same size as count matrices. Correcting (not good solution)."
    

    # calculate avg transition time
    r = np.maximum(cntTotalM, np.ones_like (cntTotalM))
    avgInterArrivalTotalM = np.divide (sumInterArrivalTotalM, r)

    # prune locks not used very much
    cntP, avgInterArrivalP, tVecP, idMap = pruneAll (cntTotalM, avgInterArrivalTotalM, servTimeVec, 1000)

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
    parser.add_argument("datafile", help="Prefix of files containing thread IDs, lock IDs and acquire timestamps, sorted by time. Name should be \"prefix_{#threads}th_{acq,rel}.dat\"")
    parser.add_argument("servfile", help="File containing lock IDs and average service time, sorted by lock ID.")


    options = parser.parse_args()

    if options.debug:
        print >> sys.stderr, "options:", options

    # we assume the data files follow the naming convention appname_{#threads}th_{acq,rel}.dat
    
    servfile = open(options.servfile)
    servTimeVec = waitingTimeParse(servfile)
    servfile.close()

    # parse data for 8 thread case
    (acqDic, relDic, create8Vec) = parseDicTuple (options.lockacq, 8)

    # parse data for 1 thread case
    (acq1Dic, rel1Dic, create1Vec) = parseDicTuple (options.lockacq, 1)

    servTimeList = map (waitingTime, acq1D.values(), rel1D.values())

    servTimeArr = np.zeros_like( max (servTimeList, key = len))
    for l in servTimeList:
        servTimeArr[0:len(l)] += l
    servTimeArr /= len (servTimeList)


    lockMap (create8Vec, create1Vec)
    

    analyze (acqDic, relDic, servTimeVec, options.numCores)

    sys.exit(0)
    # END MAIN


# uncomment when running from command line

# if __name__ == '__main__':
#    main() 
