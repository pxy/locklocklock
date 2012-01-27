"""SLAP-
$ Time-stamp: <2012-01-27 13:22:11 jonatanlinden>

README:
A collection of tools to do a queueing network analysis on sequences
of timestamps representing threads lock accesses.

datafiles should have row format 'timestamp threadID lockID', and
should be sorted by the timestamp.
"""

from collections import defaultdict
import os,re,csv,subprocess,math,struct,logging
import numpy as np
import numpy.ma as ma
import operator as op
from mva import st_mva, mva_multiclass
import histo
from itertools import *
from functools import partial




logging.basicConfig(format="%(funcName)s: %(message)s", level=logging.DEBUG)

# CONSTANTS

def enum(**enums):
    return type('Enum', (), enums)

timelines = enum(WAIT=1, QUEUE=2, SERV=3, INTER=4)


# OBJECT


class LockTrace(object):
    
    def __init__(self, tryD, acqD, relD, namesD):
        (self.tryD, self.acqD, self.relD, self.namesD) = (tryD, acqD, relD, namesD)
        self.start = min (self.start_ts())
        self.end   = max (self.end_ts())
        self.tl    = None
        self.classes = None

    def start_ts(self):
        return zip(*elem_from_list_dic(self.tryD, 0))[1]

    def end_ts(self):
        return zip(*elem_from_list_dic(self.tryD, -1))[1]

    def delete_thread(self, pos_idx):
        key = self.tryD.keys()[pos_idx]
        del self.tryD[key]
        del self.acqD[key]
        del self.relD[key]
        return key

    def set_classes(self, class_l):
        self.classes = class_l
        self.pop_vector = tuple(map (len, filter(None, class_l)))

    def analyze(self):
        if self.classes:
            r = multi_analyze(self.tryD, self.acqD, self.relD, self.namesD, self.classes)
            (self.rout_l, self.serv_rates, self.q_type, self.meas_wait, self.rout_cnt) = (r[0], r[1], r[3], r[4], r[6])
        else:
            print "Classes undefined"

    def serv_times(self):
        if self.serv_rates != None:
            return 1/self.serv_rates[1::2]
        else:
            return None

    def mva(self, pop_vector = None):
        if not self.rout_l:
            self.analyze(self)
        res_tmp = mva_multiclass(self.rout_l, self.serv_rates, pop_vector or tuple(map(len, self.classes)), self.q_type)
        self.est_wait = res_tmp[0][1::2]
        self.est_qlen = res_tmp[1][1::2]
        self.lam      = res_tmp[2]

    

    def time_line (self, kind, use_class=False):
        if   kind == timelines.INTER:
            f = self.tl_inter
        elif kind == timelines.SERV:
            f = self.tl_serv

        self.tl_class = use_class
            
        tls = map (f, self.tryD.values(), self.acqD.values(), self.relD.values(), self.tryD.keys())
        if use_class and self.classes:
            self.tl = [[] if cl == [] else sorted(merge_lists(idx(cl, tls))) for cl in self.classes]
        else:
            self.tl = sorted(merge_lists(tls))

    def tl_serv (self, startSeq, middleSeq, endSeq, tag):
        '''returns a sequential timeline containing (trytime, serv.time, threadid, lockid).
        '''
        return map (lambda (i,x): (x[1], endSeq[i][1] - middleSeq[i][1], tag, x[0]), enumerate (startSeq))

    # interarrival time timeline.
    def tl_inter (self, startSeq, middleSeq, endSeq, tag):
        return map (lambda (i,x): (x[1], x[1] - endSeq[i][1], endSeq[i][0], tag, x[0]), enumerate (startSeq[1:]))

    def num_accesses (self, lock_id, cls = None):
        if not self.tl:
            print "Define a timeline."
            return None
        if not (cls or self.tl_class):
            n_acc = len(filter(lambda x: x[-1] == lock_id, self.tl))
        elif self.tl_class and cls:
            n_acc = len(filter(lambda x: x[-1] == lock_id, self.tl[cls]))
        elif self.tl_class and not cls:
            n_acc = len(filter(lambda x: x[-1] == lock_id, merge_lists(self.tl)))
        return n_acc



# locktrace from a file
class ParsedLockTrace(LockTrace):
    def __init__(self, path):
        (tryD, acqD, relD, namesD) = parseDicTuple(path)
        super(ParsedLockTrace, self).__init__(tryD, acqD, relD, namesD)


# defines a new locktrace from a subinterval of an old one
class SubLockTrace(LockTrace):
    def __init__(self, old_lt, start_ts, end_ts):
        if not old_lt.classes:
            raise ValueError("The locktrace parameter must have classes defined")
        relD = defaultdict(list)
        tryD = defaultdict(list)
        acqD = defaultdict(list)
        for k,v in old_lt.tryD.iteritems(): #v is a list of the tuple (lockid,timstamp)
            for i,t in enumerate(v):
                if t[1] > start_ts and t[1] <= end_ts: # if access starts in (start_ts, end_ts], keep all accesses
                    tryD[k].append(t)
                    relD[k].append(old_lt.relD[k][i])
                    acqD[k].append(old_lt.acqD[k][i])
        
        super(SubLockTrace, self).__init__(dict(tryD), dict(acqD), dict(relD), old_lt.namesD)
        # map old classes to new ones
        class_map = [map(lambda x: (old_lt.tryD.keys()[x], x), cl) for cl in old_lt.classes]
        self.set_classes(filter_class(class_map, self.tryD))



def sub_lt_by_class(lt, cls):
    ls = idx(lt.classes[cls], lt.tryD.values())
    mx = max ([x[-1][1] for x in ls])
    mn = min ([x[0][1]  for x in ls])
    return SubLockTrace(lt, mn, mx)

def rate_of_lock(lt, cls, lockid):
    f0 = filter (lambda x: x[3]==lockid, lt.tl[cls])
    return len(f0)*20./float((f0[-1][0] - f0[0][0]))
    
# helper function to sublocktrace
# maps classes according to class map, based on keys in dic.
def filter_class(class_map, dic):
    l = [[t[0] for t in filter(lambda x: x[0] in dic.keys(), sublist)] for sublist in class_map]
    return [map (dic.keys().index, sublist) for sublist in l]


# returns a list of the timestamps when lock lock has been accessed splitsize times
# input timeline, lockid, number of lockaccesses of each slice
def timestamps_by_cnt_at_lock(tl, lock, splitsize):
    tlf = filter(lambda x: x[3] == lock, tl)
    l = [x[1][0] for x in filter (lambda (i,tp): i % splitsize == 0, enumerate(tlf))]
    return l[1:] if l else []

# splits a locktrace into a list of locktraces based on the timestamps in splitpoints
def split_locktrace_by_time (lt, splitpoints):
    res = []
    prev = 0
    for i in splitpoints:
        res.append(SubLockTrace(lt, prev, i))
        prev = i
    res.append(SubLockTrace(lt, prev, max(lt.end_ts())))
    return res

# input locktrace, class (that is used to determine number of accesses), chunksize, increase of number of threads (e.g., (2,2,2) -> (10,10,10) is an increase of 5)
def interval_analysis(lt, cls, lock, splitsize, inc=1):
    splits = timestamps_by_cnt_at_lock(lt.tl[cls], lock, splitsize)
    lt_list = split_locktrace_by_time (lt, splits)
    for lt_sub in lt_list:
        # individual analysis on each timeslot
        lt_sub.analyze()
        lt_sub.mva(tuple(map(partial(op.mul, inc), lt_sub.pop_vector)))
        lt_sub.time_line(timelines.SERV, use_class=True)
    return lt_list


def unpack_timeline(f_name):
    '''Unpacks a binary file containing 64-bit timestamps in lots of five, skipping any zeroes.
    '''
    tl = []
    with open(f_name) as f:
        for block in chunked(f, 40):
            a,b,c,d,e, = struct.unpack('qqqqq', block)
            if a == 0:
                continue
            tl.append((a,b,c,d,e))
    return tl

# --------------------------------------------------------------------------
# UTILS

def first_where(l, p):
    (i for i,v in enumerate(l) if p(v)).next();

def chunked (f, n):
    '''reads chunks of binary data of size n from open file f until EOF
    '''
    while 1:
        s = f.read(n)
        if not s:
            break
        yield s
 
def chunk_timeline(tl, size):
    prev = 0
    l = []
    for i in xrange(size,len(tl), size):
        l.append(tl[prev:i])
    return l
       

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

def cumulative_sum(l):
    cum_sum = []
    y = 0
    for e in l:  
        y += e
        cum_sum.append(y)
    return cum_sum

def dictToArray (dict):
    size = max(dict.iteritems())[0] + 1
    arr = np.zeros (size)
    for k,v in dict.iteritems():
        arr[k] = v
    return arr

def dictToMatrix (dict, dim = 0):
    '''Converts a two-level nested dictionary to a matrix, where the rows are given by
    the values of the outer dictionary.
    '''
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

def imax(l, key=None):
    '''Indexed max function
    '''
    return l.index(max(l, key=key))

def imin(l, key=None):
    '''Indexed min function
    '''
    return l.index(min(l, key=key))


def count(firstval=0, step=1):
    '''start, start+step, start+2*step, ...
    '''
    x = firstval
    while 1:
        yield x
        x += step

def merge_lists(lls):
    '''Flattens a list of lists.
    '''
    return [item for sublist in lls for item in sublist]


def partsOfList(l, idxsL):
    return op.itemgetter(*idxsL)(l)

def mysum(l):
    s2 = 0
    s = 0
    for e in l:
        s += e
        s2 += e * e
    return (s, s2)

def idx(idxl, l):
    if len (idxl) == 1:
        return [l[idxl[0]]]
    return op.itemgetter(*idxl)(l)


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
    in increasing order by timestamp (at least per thread id, row[1])
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


def avgWaitTime (tryLockSeq, relLockSeq, dim=0):
    """Calculates average waiting time (service time + queue time) per lock
    INPUT: tryLockSeq, relLockSeq : a tuple list of the form (lockID, timestamp)
    """
    # create a dictionary indexed by lockIDs
    timeD = waitingTime (tryLockSeq, relLockSeq)
    if dim==0:
        dim = max (timeD.keys()) + 1
    arr = np.zeros(dim)
    arr2 = np.zeros(dim)
    # the number of accesses to each lock will be defined by the length of
    # each dictionary value
    countArr = np.zeros(dim)
    for k,v in timeD.iteritems():
        countArr[k] = len(v)
        s1, s2 = mysum(v)
        arr[k] = float(s1) / len(v)
        arr2[k] = float(s2) /len(v)

    return (arr, arr2, countArr)

def servTime (acqSeqL, relSeqL, dim=0):
    servTimeList = map (partial(avgWaitTime, dim=dim), acqSeqL, relSeqL)
    servTimes, servTimesSq, counts = zip (*servTimeList)
    servTimesMtx = listOfArrToMtx (servTimes)
    servTimesSqMtx = listOfArrToMtx (servTimesSq)
    countMtx = listOfArrToMtx (counts)
    # norm of columns
    norms = normalizeRowWise(countMtx.T).T
    ma_servTimesMtx = ma.array(servTimesMtx, mask = servTimesMtx == 0)
    ma_servTimesSqMtx = ma.array(servTimesSqMtx, mask = servTimesSqMtx == 0)
    return ma.average (ma_servTimesMtx, axis=0, weights=norms), ma.average(ma_servTimesSqMtx, axis=0, weights=norms)


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
        for k,v in tryD.iteritems():
            newTryD[k] = list(dropwhile(lambda x: x[1] < start, v))
            newAcqD[k] = acqD[k][len(tryD[k]) - len(newTryD[k]):]
            newRelD[k] = relD[k][len(tryD[k]) - len(newTryD[k]):]
    return (newTryD, newAcqD, newRelD)



def doubleDimWithZeros (arr):
    newdim = len (arr)*2
    t = ma.zeros(newdim)
    for i in range(0, newdim, 2):
        t[2*i+1] = arr[i]
    return t


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


def elem_from_list_dic(tDic, idx):
    return map(lambda x: tDic[x][idx], tDic.keys())

def first_and_last_thread(tDic):
    tmax = imax(elem_from_list_dic(tDic, -1), key=op.itemgetter(1))
    tmin = imin(elem_from_list_dic(tDic,  0), key=op.itemgetter(1))
    return tDic.keys()[tmax], tDic.keys()[tmin]




# --------------------------------------------------------------------------
# entry points of application

# classL : list of lists, where each list contains the thread id's of one specific class.
def multi_analyze (tryDic, acqDic, relDic, namesVec, classL, overhead=0.0):
#generate routing, serv.time and interarrivals
    routCntL = []
    servTimeVecL = []
    servTimeSqVecL = []
    avgIAML = []
    waitTimeVecL = []
    nLocks = (max ([max (x, key=op.itemgetter(0)) for x in idx(merge_lists(classL), tryDic.values())], key=op.itemgetter(0)))[0] + 1

    for cl in filter(None, classL):
        trySeqL = idx(cl, tryDic.values())
        acqSeqL = idx(cl, acqDic.values())
        relSeqL = idx(cl, relDic.values())
        routCntL.append(routingCntMtx(trySeqL, nLocks))
        serv, servSq = servTime(acqSeqL, relSeqL, dim=nLocks)
        servTimeVecL.append(serv)
        wait, _ = servTime(trySeqL, relSeqL, dim=nLocks)
        waitTimeVecL.append(wait)
        r = np.maximum(routCntL[-1], np.ones_like (routCntL[-1]))
        sumInterArrivalM = interArrivalMtx (trySeqL, relSeqL, nLocks)
        avgIAML.append(np.divide (sumInterArrivalM, r))

    routL = map (normalizeRowWise, routCntL)
    IQs = map (insertIntermediateQs, routL, avgIAML, servTimeVecL)
    newRoutL, newServTimeVecL = zip (*IQs)

    servTimeM = listOfArrToMtx (newServTimeVecL)
    qt = list(islice(cycle((1,0)), nLocks*2))

    ma_servTimeM = ma.array(servTimeM, mask = servTimeM == 0.0)
    ma_servTimeM[1::2] = ma_servTimeM[1::2] + overhead

    # TODO FIX: divide by zero
    servTimeM2 = 1.0/ma_servTimeM

    cntPerClass = map (np.sum, routCntL)
    cntClassLock = zip(*[np.sum(x, axis=0) for x in routCntL])
    
    waitTimeVectors = np.array(map(list, zip(*[l.filled(0) for l in waitTimeVecL])))

    return newRoutL, servTimeM2.T, tuple(map(len, classL)), qt, waitTimeVectors, avgIAML, routCntL


def analyze (tryDic, acqDic, relDic, namesVec, numT,overheadF = lambda x: x):

    cntTotalM = routingCntMtx(tryDic.values())
    sumInterArrivalTotalM = interArrivalMtx (tryDic.values(), relDic.values())
    nLocks = (max ([max (x, key=op.itemgetter(0)) for x in tryDic.values()], key=op.itemgetter(0)))[0] + 1
    
    # sanity check
    if sumInterArrivalTotalM.shape[0] != cntTotalM.shape[0]:
        print "WARNING: count matrix not same size as interarrival time matrix."

    servTimeVec, servTimeVecSq = servTime (acqDic.values(), relDic.values(),dim=nLocks)

    servTimeVecWithOH = overheadF(servTimeVec)

    # calculate avg transition time
    r = np.maximum(cntTotalM, np.ones_like (cntTotalM))
    avgInterArrivalTotalM = np.divide (sumInterArrivalTotalM, r)
    rout = normalizeRowWise (cntTotalM)
    newRout, servTimes = insertIntermediateQs (rout, avgInterArrivalTotalM, servTimeVecWithOH)

    return newRout, 1/servTimes, cntTotalM

def runandprintmva (rout, servRates, numT, tryDic, relDic, namesVec, cntTotalM):
    # mva it
    T,N = st_mva (rout, servRates, numT)
    servRates = servRates.compressed()
    servTimes = 1/servRates[1::2]
    estincr  = (T*servRates)[1::2]

    cntTot = np.sum (cntTotalM, axis=0)

    # actual waiting time for numT threads
    _actualWait, _ = servTime(tryDic.values(), relDic.values())
    actualWait = _actualWait.compressed()
    
    for i,e in enumerate (T[1::2]):
        print '%s : act: %6.0f, est: %6.0f, serv: %6.0f, est.incr: %1.3f, acc: %d' % (namesVec[i], actualWait[i], T[1::2][i], servTimes[i], estincr[i], cntTot[i])

    return zip (namesVec, actualWait, T[1::2], servTimes, N[1::2])


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


def startBucket (val, bucketsize):
    return int(bucketsize * math.floor(float(val)/bucketsize))

def endBucket (val, bucketsize):
    return int (bucketsize * math.ceil (float(val)/bucketsize))


def interval(tl, start=None, end=None):
    if not (start or end):
        return None
    if end:
        tl = takewhile(lambda x: x[0] < end, tl)
    if start:
        tl = dropwhile(lambda x: x[0] < start, tl)
    return tl

def split(predicate, iterable):
    # dropwhile(lambda x: x<5, [1,4,6,4,1]) --> ([1,4],[6,4,1])
    iterable = iter(iterable)
    prefix = []
    tail = iter([])
    for x in iterable:
        if predicate(x):
            prefix.append(x)
        else:
            tail = chain([x], iterable)
            break
    return prefix, tail

# the same as before, but this time describes the relation between two groups of timestamps
def corr_lag_k2 (tl0, tl1, timestep, lag):
    # only analyse the part the sequences have in common
    st = max(tl0[0][0], tl1[0][0])
    en = min(tl0[-1][0], tl1[-1][0])
    # pad time frames to look nice
    start = startBucket(st, timestep) 
    end = endBucket(en, timestep)
    tl02 = list(interval(tl0, start=start, end=end))
    tl12 = list(interval(tl1, start=start, end=end))

    # mean number of accesses per time frame
    steps = (end - start)/float(timestep)
    mean0 = len(tl02)/steps
    mean1 = len(tl12)/steps

    ret0, ret1 = [],[]
    timeLine0,timeLine1 = tl02,tl12
    for i in range(start, end + 1, timestep):

        prefix0, timeLine0 = split(lambda x: x[0] < i, timeLine0)
        ret0.append(len(prefix0))
        #not working correctly if iterator is not evaluated at each step
        #timeLine0 = interval(timeLine0, start=i)
        #timeLine0 = list(interval(timeLine0, start=i))
        prefix1, timeLine1 = split(lambda x: x[0] < i, timeLine1)
        ret1.append(len(prefix1))
        #ret1.append(len(list(interval(timeLine1, end=i+timestep))))
        #not working correctly if iterator is not evaluated at each step
        #timeLine1 = interval(timeLine1, start=i)
        #timeLine1 = list(interval(timeLine1, start=i))

    corr, var0, var1 = 0, 0, 0
    # the inital part of the sequences should be taken into account when calc. abs. variance
    for i in range(lag):
        var0 += math.pow(ret0[i] - mean0, 2.0)
        var1 += math.pow(ret1[i] - mean1, 2.0)
        
    for i in range(lag, len(ret0)):
        # this defines correlation between arrival of t0 and an arrival of
        # sequence t1 lag*timestep timeunits later
        corr += (ret0[i - lag] - mean0)*(ret1[i] - mean1)
        var0 += math.pow(ret0[i] - mean0, 2.0)
        var1 += math.pow(ret1[i] - mean1, 2.0)

    # get means over appropriate sequences
    corr /= len(ret0) - lag 
    var0 /= len(ret0)
    var1 /= len(ret1)

    # the joint variance defined through the streams std. dev.
    vari = math.sqrt(var0*var1) 
    
    
    return corr/vari # normalised according to the absolute variance

