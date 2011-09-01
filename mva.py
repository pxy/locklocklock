#!/usr/bin/env python

import os,sys,operator,math
import numpy as np
from numpy.linalg import solve

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
