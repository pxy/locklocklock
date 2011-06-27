#!/usr/bin/env python

import os,sys,operator,math
import numpy as np
from numpy.linalg import solve

def mva(p, servrates, M):
    K = p.shape[0] #number of queues
    u = servrates
    #q_type = []
    q_type = map (lambda (i): i % 2, range(K+1))
    #for i in range(K + 1):
    #    q_type.append((i) % 2) # queue type, 0: inf, 1: 1-server

    # A K by K identify matrix to be used later to solve the traffic equation
    I = np.identity(K) 

    #substitute the last row in the transpose matrix of I-p with an array with ones
    # and try to solve that equation (normalize the ratios in teh traffic equation)
    q = np.ones(K)
    tmp = (I-p).T
    r = np.vstack((tmp[:-1,:],q)) 
    
    N = np.zeros((K,M+1)) # matrix for queue length of node K with M customers 
    W = np.zeros((K,M+1)) # response time of node K with M customers

    a = np.zeros(K) # Zero vector with the last element being 1.0
    a[K-1] = 1.0

    v = solve(r,a)

    for m in range(1,M+1): #[1..M] #loop over the number of customers
        #step 1
        for k in range(1,K+1): #loop over the queues
            W[k-1,m] = (q_type[k-1]*N[k-1,m-1] + 1.0)/u[k-1]
#        test = (q_type*N[:,m-1] + 1.0)
		#step2
        sum = np.sum (W[:,m]*v)
        lam = m/sum
        
        #step3
        N[:,m] = lam*v*W[:,m]

    # return the waiting times for the case when M customers are in the network
    return W[:,M]
