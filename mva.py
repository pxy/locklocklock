#!/usr/bin/env python

import os,sys,operator,math
import numpy as np
from numpy.linalg import solve

def mva(p, servrates, M):
    K = p.shape[0] #number of queues
    u = servrates
    q_type = []
    for i in range(M + 1):
        q_type.append((i+1) % 2) # queue type, 1: inf, 0: 1-server

    I = np.identity(K,float) #An K by K identify matrix to be used later to solve the traffic equation

    #We can subsitute the last row in the transpose matrix of I-p with an array with ones and try to solve that equation(normalize the ratios in teh traffic equation)
    q = np.ones(K,float)
    tmp = (I-p).transpose()
    r = np.vstack((tmp[:-1,:],q)) 
    
    N = np.zeros((K,M+1),float) #matrix for queue length of node K with M customers 
    W = np.zeros((K,M+1),float) #response time of node K with M customers

    a = np.zeros(K,float) #Zero vector with the last element being 1.0
	#a[K-1] = 1.0
    a[K-1] = 1.0

    v = solve(r,a) 
    print "r: ",r
    print "a: ",a
    print "Traffic equation:"
    print "v: ",v

    for m in range(1,M+1): #[1..M] #loop over the number of customers
        #step 1
        for k in range(1,K+1): #loop over the queues
            if q_type[k-1] == 0: #if the queue is as one server queue
                W[k-1,m] = (N[k-1,m-1] + 1)/u[k-1]
            else: #otherwise the queue is a infinite server queue
                W[k-1,m] = 1.0/u[k-1]
		#step2
        sum = 0.0
        for k in range(1,K+1):
            sum += W[k-1,m]*v[k-1] 
        lam = m/sum
        #step3
        for k in range(1,K+1):
            N[k-1,m] = v[k-1]*lam*W[k-1,m]

    for k in range(1,K+1): #loop over the queues
	    print "waiting time for queue ",k,": ",W[k-1,M]
    return W
