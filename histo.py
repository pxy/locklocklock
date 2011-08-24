import numpy as np
import os, csv


def normalizeList(l, sumTo=1):
    return [ x/(sum(l)*1.0)*sumTo for x in l]


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
	w = csv.writer(open(path + os.sep + name + '.dat', 'w'), delimiter=' ')
	w.writerows(data)
	
	
