#!/usr/bin/env python

# requires installing argparse package.

import os,sys,operator,math
from numpy import *
from numpy.linalg import solve

p = array([[0,1.0,0,0,0],[0,0,1,0,0],[0,0,0,0.4,0.6],[1,0,0,0,0],[1,0,0,0,0]]) #routing matrix 
K = 5 #number of queues
M = 6 #number of customers
#u = ones(K,float) #service rates of the queues
u = (1,1,1,1,1)
q_type = (1,0,1,0,0) #1 means it's an infinite server queue and 0 means it's a one server queue (we only have these two kinds in our system now)
I = identity(K,float) #An K by K identify matrix to be used later to solve the traffic equation

print "p",p
#We can subsitute the last row in the transpose matrix of I-p with an array with ones and try to solve that equation(normalize the ratios in teh traffic equation)
q = ones(K,float)
tmp = (I-p).transpose()
r = vstack((tmp[:-1,:],q)) 

N = zeros((K,M+1),float) #matrix for queue length of node K with M customers 
W = zeros((K,M+1),float) #response time of node K with M customers

a = zeros(K,float) #Zero vector with the last element being 1.0
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
		#print "W[",k-1,m,"]",W[k-1,m]
	#step2
	sum = 0.0
	for k in range(1,K+1):
		sum += W[k-1,m]*v[k-1] 
	lam = m/sum
	#step3
	for k in range(1,K+1):
		N[k-1,m] = v[k-1]*lam*W[k-1,m]
		#print "N[",k-1,m,"]",N[k-1,m]
for k in range(1,K+1): #loop over the queues
	print "waiting time for queue ",k,": ",W[k-1,M]
