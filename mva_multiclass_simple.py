#!/usr/bin/env python
import os,sys,operator,math
import numpy as np
from itertools import *
from numpy.linalg import solve
from collections import defaultdict
np.set_printoptions(threshold=np.nan)


def getPopulationVs (nClassL, exists):
    if exists[nClassL]:
        return []
    else:
        exists[nClassL] = 1
        l = chain(*[(getPopulationVs(x, exists)) for x in dependentsV(nClassL)])
        return chain(l, [nClassL])

def unitVs (dim):
    return map (tuple, np.identity(dim, dtype=int))

def dependentsV(tup):
    units = unitVs (len(tup))
    return filter (lambda x: all(i >= 0 for i in x), [tuple(np.subtract(tup,x)) for x in units])

def dependentV(tup, c):
    unit = unitVs(len(tup))[c]
    res = np.subtract(tup, unit)
    res[res<0] = 0
    return tuple(res)

#This function implements the mean value analysis for the multi-class closed network
#inputs: routL: routing matrix list
#	 servrates: service rates for the queues for different job classes, should be a #queues x # classes matrix
#	 nClassL: a list of the number of threads in each class, the length of the list is the number of classes
def mva_multiclass(routL, servrates, nClassL, queueType, vr=None):
    #total number of queues and classes
    K = servrates.shape[0] 
    n_class = len(routL)
    all_popuV = list (getPopulationVs(nClassL, defaultdict(int))) #the final population vector
    if vr != None:
        e = vr
    else:
        e = np.array(getVisitRatios(routL))

    #if np.where(e<0.0) or np.where(servrates == 0.0):
    #    print "WARNING: bad input"
    #    return -1

    e[e < 0.0] = 0.0

    
    
    #STEP 1: initialize the number of jobs matrices,
    # N is a dictionary of matrices, the key is the pop.vectors
    # Rows are the different values for the classes, hence #rows == #queues
    # the node index and r: the class index as the indices and the key of the dictionary is a population matrix
    T = {}
    N = {}
    lam = {} # the rate of each class for a specific population vector
    for k in all_popuV:
        T[k] = np.zeros((K,n_class))
        N[k] = np.zeros((K,n_class))
        lam[k] = np.zeros(n_class)

    # ***BEGIN ALGO***

    for k in all_popuV:
        #STEP 2.1
        # calculate T

        for i in range(K): # queues
            # T[k][i] = (1.0/2*servrates[i])*(1.0+A_k[i]) + (2 - 1)*(1 - (1/2)*)
            if queueType[i] == 1:#if node i is an infinite server node, the response time is just the service time
                T[k][i] =  1.0/servrates[i]
            else: #if node i is a single server queue
                # new T[k] is total service time for expected no. of cust waiting + new job
                # A_k is the total number of jobs waiting at the arrival of a new job
                A_k = np.array([N[dependentV(k, x)][i].sum() for x in range(n_class)])
                T[k][i] = (1.0/servrates[i])*(1.0+A_k) # R_ck
            T[k][i][servrates[i] == 0.0] = 0.0

        #STEP 2.2
        # calculate throughput 
        #for each class/row, sum together expected time
        sum2 = np.diag(np.dot(e, T[k]))
        lam[k] = np.array(k)/sum2

        #STEP 2.3
        #for each class and each server, update est. no. of customers in server.
        N[k] = T[k]*lam[k]*e.T # 

        print N[k]
    # ***END ALGO*** for loop over pop.vectors.

    #lam1 = e[0][2]*lam[k][0]
    #lam2 = e[1][2]*lam[k][1]
    #print "Throughput for queue ", i, " class 1: ", lam1, "class 2:",lam2, "throughput difference: ", lam1 - lam2

    U = np.zeros((K, n_class))
    for i in range(K):
        #for r in range(0,n_class):
            #print "N",k,i+1,r+1, "=", lam[k][r],"*",T[k][i][r],"*",e[r][i], "=",N[k][i][r]
        U[i,:] = 1.0/servrates[i]*e[:,i]*lam[nClassL]
    

    return  T[nClassL], N[nClassL]


def probstate (j, m, vr, servrates, lam):
    if j == 0:
        temp = (vr / servrates ) * lam 


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
	lamL, T = mva_multiclass(routL, servrates, cusL, qTypeL)
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
		lamL, T = mva_multiclass(routL, servrates, cusL, qTypeL)
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
        servrates = np.array([[1.0,1.0],[1.0,1.0],[1.0,x],[1.0,0.7]])
        la,T = mva_multiclass([rout,rout], servrates, (2,2), [1,0,1,0])
        print "lambda before adjusting rates"
        print la[0],la[1]
        print "The adjusted rates for "
        print x
        newRates = adjustRate([rout,rout], servrates, (2,2), [1,0,1,0], [3,1],la[0]/100.0)
        print newRates
        l,T = mva_multiclass([rout,rout], newRates, (2,2), [1,0,1,0])
        print l[0],l[1]


# this test should return true :)
def closedsingleclasstest():
    p_back = 0.5
    R = np.zeros((7,7))
    R[0,1] = 1.0
    R[1,2:5] = 1.0/3
    R[2:5,5:7] = (1-p_back)/2.0
    R[2:5,0] = p_back
    R[5:7,2:5] = 1/3.0
    S = np.array([0.2, 2, 1/0.8, 1/0.8, 1/0.8, 1/1.8, 1/1.8])
    correct_answer = np.array([[ 5.01121098],
       [ 0.5011211 ],
       [ 0.53452917],
       [ 0.53452917],
       [ 0.53452917],
       [ 0.90201798],
       [ 0.90201798]])
    t, n, answer = mva_multiclass([R], S, (20,), [1,0,0,0,0,0,0])
    return all(a < 0.00001 for a in np.absolute(answer - correct_answer))
    

def closedmclasstest():
    S = np.array([[1, 0.25, 0.125, 1/12.0],
                  [0.5, 0.2, 0.1, 1/16.0]])
    e = np.array([[1, 0.4, 0.4, 0.2],
                  [1, 0.4, 0.3, 0.3]])
    nPop = (1,2)

    print e
    return mva_multiclass ([], S.T, nPop, [0,0,0,1], vr = e)

def closedmclasstest2():
    S = np.array([[1,1/2.0],[1/3.0,1/4.0]])
    r = np.array([[0, 1], [1, 0]])
    nPop = (1,1)
    return mva_multiclass ([r, r], S, nPop, [0,0])


    
