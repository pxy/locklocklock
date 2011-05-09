#!/usr/bin/env python

# requires installing argparse package.

import os,sys,operator,math
from numpy import *
from numpy.linalg import solve

p = array([
	[  9.54213092e-01,   0.00000000e+00,   4.56128134e-02,
	   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	   1.74094708e-04,   0.00000000e+00],
	[  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	   0.00000000e+00,   0.00000000e+00],
	[  1.40805407e-04,   3.66094058e-02,   9.62968178e-01,
	   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	   1.40805407e-04,   1.40805407e-04],
	[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	   0.00000000e+00,   0.00000000e+00],
	[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	   0.00000000e+00,   0.00000000e+00],
	[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	   0.00000000e+00,   0.00000000e+00],
	[  7.69230769e-02,   0.00000000e+00,   0.00000000e+00,
	   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	   9.23076923e-01,   0.00000000e+00],
	[  0.00000000e+00,   5.00000000e-01,   0.00000000e+00,
	   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	   0.00000000e+00,   5.00000000e-01]]) #routing matrix


K = 8 #number of queues
M = 6 #number of customers
u = ones(K,float) #service rates of the queues
q_type = (1,0,0,0,0,0,0,0) #1 means it's an infinite server queue and 0 means it's a one server queue (we only have these two kinds in our system now)
I = identity(K,float) #An K by K identify matrix to be used later to solve the traffic equation

#def file_to_matrix(file_name,n):
	#f = open(file_name,'r')
	#for line in f:
		#x= line.replace(']','')
		#y= x.replace('[','')
		#z = y.split(',')
	#print "z",z
	#p = zeros((n,n),float)
	#for i in range(0,n):
		#for j in range(0,n):
			#print "i:",i,"j:",j
			#p[i,j] =  z[i*n+j]
	#return  p

def file_to_matrix(file_name,n): #Reads a file which contains the routing matrix to get the routing matrix
	f = open(file_name,'r')
	for line in f:
		l = eval(line)
	w = array(l)
	return w 

#p = file_to_matrix('routing8',K)
print "p",p

#We would like to solve vp = v but matrix (I-p) is singular
#We can subsitute the last row in matrix I-p with an array with ones and try to solve that equation(normalize the ratios in teh traffic equation)
q = ones(K,float)
r = vstack(((I-p)[:-1,:],q)) 

N = zeros((K,M),float)
W = zeros((K,M),float)
sum = 0.0
a = zeros(K,float)
a[K-1] = 1.0

v = solve(r,a) 
print "r:"
print r
print "a:"
print a
print "Traffic equation:"
print v

for m in range(1,M+1): #[1..M] #loop over the number of customers
	#step 1
	for k in range(1,K+1): #loop over the queues
		if q_type[k-1] == 0: #if the queue is as one server queue
			W[k-1,m-1] = (N[k-1,m-2] + 1)/u[k-1]
		else: #otherwise the queue is a infinite server queue
			W[k-1,m-1] = 1.0/u[k-1]
		print W[k-1,m-1]
	#step2
	for k in range(1,K+1):
		sum += W[k-1,m-1]*v[k-1] 
	lam = m/sum
	#step3
	for k in range(1,K+1):
		N[k-1][m-1] = v[k-1]*lam*W[k-1][m-1]
