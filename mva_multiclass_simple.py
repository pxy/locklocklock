#!/usr/bin/env python
import os,sys,operator,math
import numpy as np
from itertools import *
from numpy.linalg import solve
from collections import defaultdict
np.set_printoptions(threshold=np.nan)

#TODO: given the number of classes and maximum number of jobs in each class, return the list of all possible population vectors
def getV(l):
    zero = tuple(np.zeros(len(l), dtype=int))

    if [zero] == l:
        return l
    else:
        s = l
        for i in l:
            s = s + getV(dependentsV(i))
        return list(set(s))



#	return [(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2)]

def getPopulationVs (nClassL, exists):
    if exists[nClassL]:
        return []
    else:
        exists[nClassL] = 1
        l = chain(*[(getPopulationVs(x, exists)) for x in dependentsV(nClassL)])
        return chain(l, [nClassL])

def unitVs (dim):
    return map (tuple, np.identity(dim, dtype=int))

# def getDependentV((a,b)):
# 	if a > 0 and b > 0:
# 		return [(a-1,b),(a,b-1)]
# 	elif a > 0:
# 		return [(a-1,b)]
# 	elif b > 0:
# 		return [(a,b-1)]
# 	else:
# 		return []

def dependentsV(tup):
    units = unitVs (len(tup))
    return filter (lambda x: all(i >= 0 for i in x), map (lambda x: tuple(np.subtract(tup,x)), units))



#This function implements the mean value analysis for the multi-class closed network
#inputs: routL: routing matrix list
#	 servrates: service rates for the queues for different job classes, should be a #queues x # classes matrix
#	 nClassL: a list of the number of threads in each class, the length of the list is the number of classes
def mva_multiclass(routL, servrates, nClassL, queueType):
    K = len(servrates) #total number of queues
    n_class = len(routL) #the number of classes
    n_threads = sum (nClassL)
    all_popuV = list (getPopulationVs(tuple(nClassL), defaultdict(int))) #getV([tuple(nClassL)]) #the finial population vector
    #all_popuV = getV([tuple(nClassL)]) #the finial population vector
    e = getVisitRatios(routL)
    #print "The visit ratio:", e
    #print "The total number of queues:", K
    #print "The total number of threads:", n_threads
    #print "The total number of classes:", n_class
    print "The final population vector:", all_popuV
    #step 1: initialize the number of jobs matrix, it is a matrix of dictionary with i: the node index and r: the class index as the indices and the key of the dictionary is a population matrix
    T = np.empty([K,n_class],dtype=dict)
    N = np.empty([K,n_class],dtype=dict) #N is K in the algorithm
    lam = np.empty([n_class],dtype=dict)
    for i in range(0,K):
        for r in range(0,n_class):
            T[i][r] = {}
            N[i][r] = {}
            for k in all_popuV:
                T[i][r][k] = 0.0
                N[i][r][k] = 0.0

    for r in range(0,n_class):
        lam[r] = {}
        for k in all_popuV:
            lam[r][k] = 0.0


    #step 2.1
    for k in all_popuV:
    #all_popuV.remove((0,0))
    #for k in all_popuV:
	#print "step 2.1 for population vector ", k
        for i in range(0,K):
            for r in range(0,n_class):
                if queueType[i] == 1:#if node i is an infinite server node, the response time is just the service time	
                    T[i][r][k] =  1.0/servrates[i][r]
                else: #if node i is a single server queue
                    sum_less = 0.0
                    for s in range(0,n_class): #from 0 to the number of classes
                        l = dependentsV(k)
                        for z in l:
                            sum_less += N[i][s][z]
                            
                    T[i][r][k] = (1.0/servrates[i][r])*(1.0+sum_less)
            #print "T[",i+1,"][",r+1,"][",k, "] = ", T[i][r][k], "1/u: ", 1.0/servrates[i][r], " sum_less: ", sum_less
    #step 2.2
        for r in range(0,n_class):
            sum2 = 0.0
            for i in range(0,K):
                sum2 += e[r][i]*T[i][r][k]
			#print "e[",i+1,"]",e[i],"T[",i+1,"]","[",r+1,"]","[",k,"]",T[i][r][k],"print sum2 in 2.2:", sum2
            lam[r][k] = k[r]/sum2
		#print "lam",r+1,k, "=", lam[r][k]
		#print the throughput of each class for each queue
        if k == tuple(nClassL):
		#for i in range(0,K): it's enough to print the info for one queue
            lam1 = e[0][2]*lam[0][k]
            lam2 = e[1][2]*lam[1][k]
		#print "Throughput for queue ", i, " class 1: ", lam1, "class 2:",lam2, "throughput difference: ", lam1 - lam2
		#print lam1, lam2, "throughput difference: ", lam1 - lam2
    #step 2.3
        for i in range(0,K):
            for r in range(0,n_class):
                N[i][r][k] = lam[r][k]*T[i][r][k]*e[r][i]
        if k == tuple(nClassL):
            for i in range(0,K):
                for r in range(0,n_class):
                    print "N",i+1,r+1,k, "=", lam[r][k],"*",T[i][r][k],"*",e[r][i], "=",N[i][r][k]
            return  [lam1,lam2]


def getVisitRatios(routL):
    return map(getVisitRatio, routL)

def getVisitRatio(p):
    K = len(p) #number of queues
    # A K by K identify matrix to be used later to solve the traffic equation
    I = np.identity(K)
    #substitute the last row in the transpose matrix of I-p with an array with ones
    # and try to solve that equation (normalize the ratios in teh traffic equation)
    q = np.ones(K)
    tmp = (I-p).T
    r = np.vstack((tmp[:-1,:],q))

    a = np.zeros(K) # Zero vector with the last element being 1.0
    a[K-1] = 1.0
    v = solve(r,a)
    return v


def updateRates(ratesL, adjustIndex, newVal):
	newL = ratesL[:]
	newL[adjustIndex[0]][adjustIndex[1]] = newVal
	return newL
		



def adjustRate(routL, servrates, cusL, qTypeL, adjustIndex, maxError):
	#print "In adjustRate:"
	rateToAdjust = servrates[adjustIndex[0]][adjustIndex[1]]
	lamL = mva_multiclass(routL, servrates, cusL, qTypeL)
	v = 0.1
	diff = lamL[0] - lamL[1]
	while math.fabs(diff) > maxError:
		#print "oldServrates:"
		#print servrates
		#print "lambda difference:"
		#print lamL[0] - lamL[1]
		if diff > 0: #(the writer is faster than the reader)
			rateToAdjust += v#increase the rate of the reader
		else:
			rateToAdjust -= v#decrease the rate of the reader
		#plug ratetoAdjust back to the rates
		servrates = updateRates(servrates, adjustIndex, rateToAdjust)
		lamL = mva_multiclass(routL, servrates, cusL, qTypeL)
		#print "newServrates:"
		#print servrates
		v = v/2
		if diff == lamL[0] - lamL[1]:
			break
		diff = lamL[0] - lamL[1]
	return servrates

def runmepan(msg):
    if msg != 'imhappy':
        return 0
    rout = [[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]]
    for x in range(10,11):
        #print "service rate for the reader local computation: ",x
        servrates = [[1.0,1.0],[1.0,1.0],[1.0,x],[1.0,0.7]]
        la = mva_multiclass([rout,rout], servrates, [2,2], [1,0,1,0])
        print "lambda before adjusting rates"
        print la[0],la[1]
        print "The adjusted rates for "
        print x
        newRates = adjustRate([rout,rout], servrates, [2,2], [1,0,1,0], [3,1],la[0]/100.0)
        print newRates
        l = mva_multiclass([rout,rout], newRates, [2,2], [1,0,1,0])
        print l[0],l[1]
