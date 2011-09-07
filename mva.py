#!/usr/bin/env python

import os,sys,operator,math
import numpy as np
from numpy.linalg import solve
from itertools import *

# STANDARD MVA

def mva(p, u, M):
    K = p.shape[0] #number of queues

    # we assume every second queue is infinite server (i.e., think time)
    q_type = np.array(map (lambda (i): i % 2, range(K)))

    v = getVisitRatio(p)

    N = np.zeros((K,M+1)) # matrix for queue length of node K with M customers 
    W = np.zeros((K,M+1)) # response time of node K with M customers

    for m in range(1,M+1): #[1..M] #loop over the number of customers
        # first column (m) is always zero

        #step 1
        # for each queue, update the waiting time
        W[:,m] = (q_type*N[:,m-1] + 1.0)/u

		#step2
        lam = m/np.dot (W[:,m],v)

        #step3
        N[:,m] = lam*v*W[:,m]

    # return the waiting times for the case when M customers are in the network
    return W[:,M]


# MULTICLASS MVA

#This function implements the mean value analysis for the multi-class closed network
#inputs: routL: routing matrix list
#	 servrates: service rates for the queues for different job classes, should be a #queues x # classes matrix
#	 nClassL: a list of the number of threads in each class, the length of the list is the number of classes
def mva_multiclass(routL, servrates, nClassL, queueType, vr=None):
    #total number of queues and classes
    K = len(servrates)
    n_class = len(routL)
    all_popuV = getPopulationVs(nClassL)
    if vr != None:
        e = vr
    else:
        e = np.array(getVisitRatios(routL))

    #STEP 1: initialize the number of jobs matrices,
    # N and T are dictionaries of matrices, the keys are the pop.vectors
    # Rows are the different values for the classes, hence #rows == #queues
    # Columns are the classes, hence index [k][q,c] is the value for queue q and class c,
    # when the population for each class is specified by k
    # lam is throughput
    T = {}
    N = {}
    lam = {} 
    for k in all_popuV:
        T[k] = np.zeros((K,n_class))
        N[k] = np.zeros((K,n_class))
        lam[k] = np.zeros(n_class)
    # ***BEGIN ALGO***

    for k in all_popuV:
        #STEP 2.1
        # calculate T
        for i in range(K): # queues
            if queueType[i] == 1:#if node i is an infinite server node, the response time is just the service time
                T[k][i] =  1.0/servrates[i]
            else: #if node i is a single server queue
                # new T[k] is total service time for expected no. of cust waiting + new job
                # A_k is the total number of jobs waiting at the arrival of a new job
                # updated, according to intuition
                A_k = np.array([(N[dependentV(k, x)][i]*(1.0/servrates[i])).sum() for x in range(n_class)])
                T[k][i] = (1.0/servrates[i] + A_k) # R_ck
                

        #STEP 2.2
        # calculate throughput 
        #for each class/row, sum together expected time
        sum2 = np.diag(np.dot(e, T[k]))
        lam[k] = np.array(k)/sum2
        
        #STEP 2.3
        #for each class and each server, update est. no. of customers in server.
        N[k] = T[k]*lam[k]*e.T #
        
    # ***END ALGO*** for loop over pop.vectors.

    return  T[nClassL], N[nClassL]


# ************************* HELPER FUNCTIONS ************************************

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

def getPopulationVs (nClassL):
    return list(product(*[range(i+1) for i in nClassL]))


def unitVs (dim):
    return map (tuple, np.identity(dim, dtype=int))

def dependentV(tup, c):
    unit = unitVs(len(tup))[c]
    res = np.subtract(tup, unit)
    res[res<0] = 0
    return tuple(res)


# *************************** TEST CASES *****************************************

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
    correct_answer = np.array([[  5.        ],
                               [  0.97950777],
                               [  1.67066031],
                               [  1.67066031],
                               [  1.67066031],
                               [ 10.63442802],
                               [ 10.63442802]])
    t,n = mva_multiclass([R], S, (20,), [1,0,0,0,0,0,0])
    return np.alltrue(np.absolute(t - correct_answer) < 0.00001)
    

# this one is not verified
def closedmclasstest1():
    S = np.array([[1, 0.25, 0.125, 1/12.0],
                  [0.5, 0.2, 0.1, 1/16.0]])
    e = np.array([[1, 0.4, 0.4, 0.2],
                  [1, 0.4, 0.3, 0.3]])
    nPop = (1,2)
    return mva_multiclass ([], S.T, nPop, [0,0,0,1], vr = e)

def closedmclasstest2():
    S = np.array([[1,1/2.0],[1/3.0,1/4.0]])
    r = np.array([[0, 1], [1, 0]])
    nPop = (1,1)
    correct_answer = np.array([[ 1.33333333,  2.5       ],
                               [ 5.        ,  7.        ]])


    t, n = mva_multiclass ([r, r], S, nPop, [0,0])
    return np.alltrue(np.absolute(t - correct_answer) < 0.00001)
