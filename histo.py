from collections import defaultdict
import numpy as np
import os, csv, string
from itertools import *
import operator as op
import math

def count(firstval=0, step=1):
    '''start, start+step, start+2*step, ...
    '''
    x = firstval
    while 1:
        yield x
        x += step


def normalizeList(l, sumTo=1):
    return [ x/(sum(l)*1.0)*sumTo for x in l]

def has_next(it):
    # the pythonian way of peeking at an iterator
    try:
        first = it.next()
    except StopIteration:
        return False
    else:
        it = chain([first], iter)
        return True


def histogram (waitSeq, bucketSize):
	arr = np.array(waitSeq)
	ma = np.amax (arr)
	print ma
	numBs = ma / bucketSize + 1
	hi = numBs * bucketSize
	return np.histogram (arr, bins=numBs, range=(0,hi))

def prunehisto (nparr):
	if nparr.shape[0] <= 600:
		retarr = nparr
	else:
		retarr = nparr[0:600]
		
	while retarr.shape[0] > 0 and retarr[-1] <= 1:
		retarr = np.delete (retarr, -1)
	return retarr

def histos (acqLockSeq, relLockSeq, bs):
	waitingTimes = map (waitingTime, acqLockSeq.values(), relLockSeq.values())
	lockDict = collapseLevel (waitingTimes)
	return map (lambda (id,l): (id, prunehisto(histogram (l, bs)[0])), lockDict.iteritems())

def writeHistos (histList, path):
	for (lid, hist) in histList:
		fd = open (path + os.sep + str(lid) + ".dat", 'w')
		hist.tofile(fd, sep='\n', format="%d")

def writeHisto (hist, name, path):
	fd = open (path + os.sep + name + ".dat", 'w')
	fd.write ('# ' + name)
	hist.tofile(fd, sep='\n', format="%d")

def writeHisto2 (hist, name, path):
	fd = open (path + os.sep + name + ".dat", 'w')
	fd.write ('# ' + name + '\n')
	np.savetxt(fd, hist, delimiter=' ', fmt='%d')

def writeResult (name, path, data):
	w = csv.writer(open(path + os.sep + name + '.dat', 'w'), delimiter=' ', lineterminator='\n')
	w.writerows(data)

def maxLockdist (timeline):
    maxval = 0
    pos = 0
    for i, t in enumerate (timeline[1:], 1):
        if t[0] - timeline[i-1][0] > maxval:
            maxval = t[0] - timeline[i-1][0]
            pos = i
    return (maxval, pos)


def mysum(l):
    s2 = 0
    s = 0
    for e in l:
        s += e
        s2 += e * e
    return (s, s2)

def variance (timeline):
    return mysum([x[2]] for x in timeline)

def timeLineSeq (startSeq, endSeq):
    return map (lambda (i,x): (x[1], endSeq[i][1] - x[1], x[0]), enumerate (startSeq))

def timeLineInterSeq (startSeq, endSeq, tag):
    return map (lambda (i,x): (x[1], x[1] - endSeq[i][1], tag, endSeq[i][0]), enumerate (startSeq[1:]))

def timeLineSeq2 (startSeq, middleSeq, endSeq, tag):
    return map (lambda (i,x): (x[1], endSeq[i][1] - x[1], endSeq[i][1] - middleSeq[i][1], tag, x[0]), enumerate (startSeq))

def timeLineSeq3 (startSeq, middleSeq, endSeq, tag):
    return map (lambda (i,x): (x[1], middleSeq[i][1],endSeq[i][1], tag, x[0]), enumerate (startSeq))

def partitionCount (tl, partitionStrategy=lambda x: x[1]):
    tl2 = sorted(tl, key=partitionStrategy)
    counts = []
    subgroup = groupby(tl2, key=partitionStrategy)
    for k, g in subgroup:
        counts.append((k, len(list(g))))
    return counts

def printPartitionCount (lcl, size, start=-1, end=-1):
    histo = np.zeros((size, len(lcl)))
    for i, cl in enumerate(lcl):
        for ct in cl[1]:
            histo[ct[0], i] = ct[1]
    return histo,lcl[0][0],lcl[-1][0]

def setLength (l, _start, _end, bucketsize, val):
    begin = l[0][0]
    pref = []
    suff = []
    if (_start < begin):
        pref = list(takewhile(lambda x: x[0] < begin, izip (count (startBucket(_start, bucketsize), bucketsize), repeat(val))))
    lend = l[-1][0] 
    if (lend < _end):
        suff = list(takewhile(lambda x: x[0] < _end, izip (count (endBucket(lend + bucketsize, bucketsize), bucketsize), repeat(val))))
    pref.append('')
    pref[-1:] = l
    pref.append('')
    pref[-1:] = suff
    return pref

def startBucket (val, bucketsize):
    return int(bucketsize * math.floor(float(val)/bucketsize))

def endBucket (val, bucketsize):
    return int (bucketsize * math.ceil (float(val)/bucketsize))


def avgTimeLineSeq (timeLines, timestep, end=0, aggr=lambda tl: sum(zip(*tl)[1])/len(tl) if tl else 0):
    timeLine = sorted(timeLines)
    start = endBucket(timeLine[0][0], timestep)
    print start
    print timeLine[0][0]
    if end != 0:
        timeLine = takewhile(lambda x: x[0] < end, timeLine)
    ret = []
    for i in count(start, timestep):
        onestep = list(takewhile(lambda x: x[0] < i, timeLine))
        timeLine = dropwhile(lambda x: x[0] < i, timeLine)
        ret.append((i, aggr(onestep)))
        if not has_next(iter):
            break
    return ret

# sorted_tl, list of arrival timestamps of locks
def autocorr_lag_k (sorted_tl, bucket, lag, aggr=lambda tl: sum(zip(*tl)[1])/len(tl) if tl else 0):
    start = endBucket(timeLine[0][0], timestep)
    if end != 0:
        timeLine = takewhile(lambda x: x[0] < end, timeLine)
    ret = []
    for i in count(start, timestep):
        onestep = list(takewhile(lambda x: x[0] < i, timeLine))
        timeLine = dropwhile(lambda x: x[0] < i, timeLine)
        ret.append((i, aggr(onestep)))
        if not has_next(iter):
            break
    return ret

def waittimecorr(timelinesL, n):
    # use first timeline as pivot

    timelinesSorted = [sorted(l, key=lambda x: x[1], reverse=True) for l in timelinesL]

    
    onetime = sorted(chain(*[islice(l, n) for l in timelinesSorted]))
    return onetime


def sliceoftimeline(timelines, start, dur):
    end = start + dur
    timelines_slice = mergeLists([list(dropwhile(lambda x: x[0] < start, takewhile(lambda x: x[0] < end, l))) for l in timelines])
    return partitionCount(timelines_slice, partitionStrategy=lambda x: x[4])

def mergelists(lls):
    '''Flattens a list of lists.
    '''
    return [item for sublist in lls for item in sublist]

def idx(idx):
    return op.itemgetter(idx)
    
class bcolors:
    TRY = '\033[95m'
    ACQ = '\033[94m'
    REL = '\033[92m'
    TIME = '\033[93m'
    EMPTY = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.TRY = ''
        self.ACQ = ''
        self.REL = ''
        self.TIME = ''
        self.EMPTY = ''
        self.ENDC = ''


def flattentupL(tup_l):
    l = []
    for tup in tup_l:
        l.extend(zip(tup[0:3], [tup[3],tup[3],tup[3]], [tup[4], tup[4], tup[4]]))
    return l

def realtimeline(timelines):
    flats = map (flattentupL, timelines)
    return sorted(mergelists (flats))

def color (st, i):
    cL = [bcolors.TRY, bcolors.ACQ, bcolors.REL]
    return cL[i] + st + bcolors.ENDC

def printTimeLine (timelines, timestep):
    ofs = "            "
    ver = bcolors.TIME + "    |       " + bcolors.ENDC
    line = [ofs, ofs, ofs, ofs]
    cnts = [0, 0, 0, 0]
    now = timelines[0][0]
    owner = -1
    for (t,i) in timelines:
        oline = line[:] 
        while now + timestep < t:
            now += timestep
            if owner >= 0 and cnts[owner] == 2:
                oline[owner] = ver
                print string.join(oline, "")
            if all (x < 2 for x in cnts):
                print bcolors.EMPTY + "--" + bcolors.ENDC

        oline = line[:]
        oline[i] = color(("%d    " % t)[8:], cnts[i])
        if owner >= 0 and cnts[owner] == 2 and i != owner:
             oline[owner] = ver

        cnts[i] = (cnts[i] + 1) % 3
        if cnts[i] == 2:
            owner = i
        now = t
        print string.join(oline, "")


def mean (l):
    return float(sum(l))/len(l)

# create a list of bursts counts, consecutive lock accesses by
# thread with idx1 after thread with idx0 has first tried to
# take it.
def cntBursts (timelines, idx0, idx1):
    sm = []
    mxval = 0
    mxidx = -1
    it = timelines[idx1]
    cntx1 = 0
    ln1 = len (it)
    for t in timelines[idx0]:
        _cnt = 0
        if not cntx1 < ln1:
            break
        while cntx1 < ln1 and it[cntx1][0] < t[0]:
            cntx1 += 1
        while cntx1 < ln1 and it[cntx1][1] < t[1]:
            cntx1 += 1
            _cnt += 1
        if _cnt > 0:
            sm.append(_cnt)
            if _cnt > mxval:
                mxval = _cnt
                mxidx = cntx1
    return sm, mxidx

def timeline_id(tryD, acqD, relD, tid_l):
    return realtimeline(map (timeLineSeq3, tryD.values(), acqD.values(), relD.values(), tid_l))

def cnt_unfair (merged_tls, n_threads):
    '''Counts the number of unfair lock handovers.

    Args:
    merged_tls -- A list of tuples (timestamp,tid), representing
    the point of time of a change of state of the thread in relation
    to the lock (try, acquire, release)
    '''
    cnts = np.zeros(n_threads)
    lids = np.array(list(repeat(-1, n_threads)))
    last = {}
    fair   = 0
    unfair = 0

    for (t,i,l) in merged_tls:
        cnts[i] = (cnts[i] + 1) % 3

        lids[i] = l
        # if we have grabbed the lock

        if cnts[i] == 2:
            # was it fair or not?
            if l in last and last[l] == i and sum(wl == l for wl in lids) > 1:
                unfair += 1
            else:
                fair += 1
        last[l] = i
        if cnts[i] == 0: # 
            lids[i] = -1
            # reduce space reqs when there is a lot of locks
            if sum(wl == l for wl in lids) < 1:
                del (last[l])
        
    return (fair, unfair)
    

#def cov(intermeans, inters):
#    n = len(inters)
#    k = len(intermeans)
#    cov = np.zeros((k,k))
#    for (i,j) in ndenumerate(cov)
#    for t, c in inters:
#        cov[i,j] += t - intermeans[l]



            
def gen_plot_wait_serv(path, name, _serv, est_wait, act_wait, class_l):
# each input is 
    serv = _serv.filled(0)
    w = csv.writer(open(path + os.sep + name + '.dat', 'w'), delimiter=' ', lineterminator='\n')
    
    w.writerow("# act_queue est_queue serv_t")

    for i in class_l:
        w.writerow("# class %d" % i)
        w.writerows(zip(act_wait[:,i] - serv[:,i], est_wait[:,i] - serv[:,i], serv[:,i]))
        w.writerows(['',''])
    
    

        

