"""SLAP-
$ Time-stamp: <2011-09-06 14:42:50 jonatanlinden>

README:
A collection of tools to do a queueing network analysis on sequences
of timestamps representing threads lock accesses.

datafiles should have row format 'timestamp threadID lockID', and
should be sorted by the timestamp.
"""

from collections import defaultdict
import os,re,csv,subprocess,math
import numpy as np
import numpy.ma as ma
import operator as op
from mva import mva, mva_multiclass
import histo
from itertools import *

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

def mergeDict(d1, d2, op=lambda x,y:x+y):
    res = defaultdict(int,d1)
    for k,v in d2.iteritems():
            res[k] = op(res[k], v)
    return res

def dictToArray (dict):
    size = max(dict.iteritems())[0] + 1
    arr = np.zeros (size)
    for k,v in dict.iteritems():
        arr[k] = v
    return arr

def dictToMatrix (dict, dim = 0):
    if dim == 0:
        dim = findGr8estKey (dict) + 1
    mtx = np.zeros ((dim, dim))
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

def imax(l, _key=None):
    '''Indexed max function
    '''
    return l.index(max(l, key=_key))

def count(firstval=0, step=1):
    '''start, start+step, start+2*step, ...
    '''
    x = firstval
    while 1:
        yield x
        x += step

def mergeLists(lls):
    '''Flattens a list of lists.
    '''
    return [item for sublist in lls for item in sublist]


def partsOfList(l, idxsL):
    return op.itemgetter(*idxsL)(l)


# end UTILS
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





def countMtxFromSeq(lockSeq, dim):
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
    return dictToMatrix(lockD, dim)






# FIX NOW !
    # transition time is:
    # (lockID_0, trylock_0), find following (lockID_0, relLock_0)
    # subtract relLock_0 from trylock_1, as in (lockID_0, trylock_1)
def timedTransitions (lockSeq, relLockSeq):
    """Calculates average transition time between locks
    """
    timeD = defaultdict(lambda : defaultdict(int))
    for i, lID in enumerate(lockSeq[1:]):
        timeD[relLockSeq[i][0]][lID[0]] += lID[1] - relLockSeq[i][1]
    return timeD


    # waiting time is:
    # (lockID_0, trylock_0), find following (lockID_0, relLock_0) in rel
    # subtract tryLock_0 from relLock_0
def waitingTime (trySeq, relSeq):
    '''POST: Dictionary containing all individual waiting times for each lock
    '''
    timeD = defaultdict(list)
    for i, tryL in enumerate(trySeq):
        rel = relSeq[i]
        if tryL[0] != rel[0]:
            print "ERROR: lock sequences not synced"
        timeD[rel[0]].append(rel[1] - tryL[1])
    return timeD


def sumWaitingTime (trySeq, relSeq):
    sum (map (op.sub, relSeq, trySeq))
    sumWait = 0
    for i, tryL in enumerate(trySeq):
        rel = relSeq[i]
        if tryL[0] != rel[0]:
            print "ERROR: lock sequences not synced"
        sumWait += rel[1] - tryL[1]
    return sumWait


def accessCntVec (lockSeq):
    countD = defaultdict(int)
    for l in lockSeq:
        countD[l[0]] += 1
    return dictToArray(countD)


def sumPredictedWaitingTime (waitVec, countVec):
    return sum (map (operator.mul, waitVec, countVec))


def totalTimeWasted (anyDic, waitTVec, servTVec):
    lockCntLst = map (accessCntVec, anyDic.itervalues())
    totCount = reduce (np.add, lockCntLst, {})
    queuingT = map (op.sub, waitTVec, servTVec)
    waitSum = {}
    for k,v in totCount.iteritems():
        waitSum[k] = queuingT[k] * v
    return sum (waitSum.itervalues())


def avgWaitTime (tryLockSeq, relLockSeq, perc=100, dim=0):
    """Calculates average waiting time (service time + queue time) per lock
    INPUT: tryLockSeq, relLockSeq : a tuple list of the form (lockID, timestamp)
    """
    # create a dictionary indexed by lockIDs
    timeD = waitingTime (tryLockSeq, relLockSeq)
    if dim==0:
        dim = max (timeD.keys()) + 1
    arr = np.zeros(dim)
    # the number of accesses to each lock will be defined by the length of
    # each dictionary value
    countArr = np.zeros(dim)
    for k,v in timeD.iteritems():
        countArr[k] = len(v)
        v.sort()
        hi = perc*len(v)/100
        arr[k] = float(sum(v[0:hi])) / (hi)
    return (arr, countArr)


def servTime (acqSeqL, relSeqL, perc=100, dim=0):
    servTimeList = map (lambda x,y: avgWaitTime(x,y, perc, dim), acqSeqL, relSeqL)
    servTimes, counts = zip (*servTimeList)
    servTimesMtx = listOfArrToMtx (servTimes)
    countMtx = listOfArrToMtx (counts)
    # norm of columns
    norms = normalizeRowWise(countMtx.T).T
    ma_servTimesMtx = ma.array(servTimesMtx, mask = servTimesMtx == 0)
    return ma.average (ma_servTimesMtx, axis=0, weights=norms)


def sumTimeMtx (trySeq, relSeq, dim = 0):
    return dictToMatrix (timedTransitions (trySeq, relSeq), dim)

def sliceSeqs (tryD, acqD, relD, start=0, end=0):
    newTryD = {}
    newAcqD = {}
    newRelD = {}
    if end != 0:
        for k,v in tryD.iteritems():
            newTryD[k] = list(takewhile(lambda x: x[1] < end, v))
            newAcqD[k] = acqD[k][0:len(newTryD[k])]
            newRelD[k] = relD[k][0:len(newTryD[k])]
    if start != 0:
        raise NotImplementedError
    return (newTryD, newAcqD, newRelD)




def insertIntermediateQs (rMatrix, tMatrix, tArray):
    '''Inserts intermediate queues representing interarrival times into a
    routingmatrix and a servicetime array.

    supposing that rMatrix is a routing matrix and tMatrix is a matrix with
    the interarrival times between the locks.
    '''

    if (rMatrix.shape[0] != len(tArray)):
        print "WARNING: Routing matrix differ from size of service time vector" 
    
    # create output matrices (twice as big)
    size = rMatrix.shape[0]
    r = np.zeros(shape=(size*2, size*2))
    t = ma.zeros(size*2)


    for i,row in enumerate(rMatrix):
        r[2*i,2*i+1]  = 1 # the queue representing the interarrival time always routs into the lock q
        r[(2*i+1),::2] = row # displace each routing by 1

    for i,col in enumerate(tMatrix.T):
        # first col of tMatrix contains the interarrival times between each lock
        # and lock 1
        if np.sum(rMatrix.T[i]) == 0.0:
            t[2*i] = 0.0
        else:
            t[2*i] = np.average(col, weights=rMatrix.T[i]) # aggregate it. (in some way)
        t[2*i+1] = tArray[i]
    return r, t




def routingCntMtx (anyL, dim = 0):
    '''A matrix with the number of transitions between locks for each thread
    '''
    return sumMatrices ([countMtxFromSeq(x, dim) for x in anyL])

def interArrivalMtx (tryDic, relDic, dim=0):
    return sumMatrices ([sumTimeMtx(x, y, dim) for (x,y) in zip(tryDic, relDic)])






# --------------------------------------------------------------------------
# entry points of application


def multi_analyze (tryDic, acqDic, relDic, namesVec, classL):
#generate routing, serv.time and interarrivals
    routCntL = []
    servTimeVecL = []
    avgIAML = []
    waitTimeVecL = []
    nLocks = 1
    for cl in classL:
        print cl
        trySeqL = [op.itemgetter(*cl)(tryDic.values())]
        acqSeqL = [op.itemgetter(*cl)(acqDic.values())]
        relSeqL = [op.itemgetter(*cl)(relDic.values())]
        routCntL.append(routingCntMtx(trySeqL, nLocks))
        servTimeVecL.append(servTime(acqSeqL, relSeqL, dim=nLocks))
        waitTimeVecL.append(servTime(trySeqL, relSeqL, dim=nLocks))
        r = np.maximum(routCntL[-1], np.ones_like (routCntL[-1]))
        sumInterArrivalM = interArrivalMtx (trySeqL, relSeqL, nLocks)
        avgIAML.append(np.divide (sumInterArrivalM, r))

    routL = map (normalizeRowWise, routCntL)
    IQs = map (insertIntermediateQs, routL, avgIAML, servTimeVecL)
    newRoutL, newServTimeVecL = zip (*IQs)
    
    # a class/serv.time.vector matrix
    servTimeM = np.zeros((len(classL), len(newServTimeVecL[0])))
    for i, x in enumerate (newServTimeVecL):
        servTimeM[i] = x
    qt = list(islice(cycle((1,0)), nLocks*2))

    ma_servTimeM = ma.array(servTimeM, mask = servTimeM == 0.0)
    servTimeM2 = 1.0/ma_servTimeM

    cntPerClass = map (np.sum, routCntL)

    return newRoutL, servTimeM2.T, tuple(map(len, classL)), qt, zip(*map(ma.getdata, waitTimeVecL)), avgIAML, cntPerClass


def analyze (tryDic, acqDic, relDic, namesVec, numT, smoothing, overheadF = lambda x: x):

    cntTotalM = routingCntMtx(tryDic.values())
    sumInterArrivalTotalM = interArrivalMtx (tryDic.values(), relDic.values())

    # sanity check
    if sumInterArrivalTotalM.shape[0] != cntTotalM.shape[0]:
        print "WARNING: count matrix not same size as interarrival time matrix."

    servTimeVec = servTime (acqDic.values(), relDic.values(), 100 - smoothing)
    servTimeVecWithOH = overheadF(servTimeVec)

    # calculate avg transition time
    r = np.maximum(cntTotalM, np.ones_like (cntTotalM))
    avgInterArrivalTotalM = np.divide (sumInterArrivalTotalM, r)
    rout = normalizeRowWise (cntTotalM)
    newRout, servTimes = insertIntermediateQs (rout, avgInterArrivalTotalM, servTimeVecWithOH)

    return newRout, 1/servTimes, cntTotalM

def runandprintmva (rout, servRates, numT, tryDic, relDic, namesVec, cntTotalM):
    # mva it
    estimate = mva (rout, servRates, numT)
    servTimes = 1/servRates[1::2]
    estincr  = (estimate*servRates)[1::2]

    cntTot = np.sum (cntTotalM, axis=0)

    # actual waiting time for numT threads
    actualWait = servTime(tryDic.values(), relDic.values())

    for i,e in enumerate (estimate[1::2]):
        print '%s : act: %6.0f, est: %6.0f, serv: %6.0f, est.incr: %1.3f, acc: %d' % (namesVec[i], actualWait[i], estimate[1::2][i], servTimes[i], estincr[i], cntTot[i])

    return zip (namesVec, actualWait, estimate[1::2], servTimes)









# ********************************************************************************
# Currently not in use



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
    return servD1, mapp

def pruneAll (rMtx, tMtx, tVec, epsilon):
    '''will prune the input matrices keeping the columns and rows for which
    predicate returns true (on either the column or the row)
    The predicate should have type numpy.ndarray -> bool (or whatever)
    '''
    f = pruneFilter (rMtx, pruneP (epsilon))
    ids = np.compress(f, np.arange(len (f)))
    return prune(rMtx, f), prune(tMtx, f), prune(tVec, f), ids

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


def parseInstrList (instrFile):
    '''Input file should be on the format "lID randomcrap [hexaddr]"
    '''
    regex = re.compile("\[(?P<hexaddr>\w+)\]")
    res = []
    recReader = csv.reader (instrFile, delimiter=' ', skipinitialspace=True)
    for _row in recReader:
        res.append(regex.search(_row[2]).group('hexaddr'))
    return res



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
