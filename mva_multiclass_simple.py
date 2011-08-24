#!/usr/bin/env python
import os,sys,operator,math
import numpy as np
from numpy.linalg import solve
np.set_printoptions(threshold=np.nan)

#TODO: given the number of classes and maximum number of jobs in each class, return the list of all possible population vectors
#def genAllPopuV():
#	return [(0,0),(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2)]
def getV(l):
	if [(0,0)] == l:
		return l
	else:
		for i in l:
			l = l + getV(getDependentV(i))
		s =  set(l)
		return list(s)

def getPopuV():
	return [(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2)]

def getDependentV((a,b)):
	if a > 0 and b > 0:
		return [(a-1,b),(a,b-1)]
	elif a > 0:
		return [(a-1,b)]
	elif b > 0:
		return [(a,b-1)]
	else:
		return []

#This function implements the mean value analysis for the multi-class closed network
#inputs: routL: routing matrix list
#	 servrates: service rates for the queues for different job classes, should be a #queues x # classes matrix
#	 nClassL: a list of the number of threads in each class, the length of the list is the number of classes
def mva_multiclass(routL, servrates, nClassL, queueType):
    K = len(servrates) #total number of queues
    n_class = len(routL) #the number of classes
    n_threads = 0 ###get the number of threads 
    for i in nClassL: #get the total number of threads
	n_threads += i
    #all_popuV = genAllPopuV() #the finial population vector
    all_popuV = getV([(nClassL[0],nClassL[1])]) #the finial population vector
    #all_popuV = getV([(2,2)]) #the finial population vector
    e = getVisitRatios(routL)
    #print "The visit ratio:", e
    #print "The total number of queues:", K
    #print "The total number of threads:", n_threads
    #print "The total number of classes:", n_class
    #print "The final population vector:", all_popuV

    #step 1: initialize the number of jobs matrix, it is a matrix of dictionary with i: the node index and r: the class index as the indices and the key of the dictionary is a population matrix
    T = np.empty([K,n_class],dtype=dict)
    N = np.empty([K,n_class],dtype=dict) #N is K in the algorithm
    lam = np.empty([n_class],dtype=dict)
    for i in range(0,K):
    	for r in range(0,n_class):
    		T[i][r] = {}
    		N[i][r] = {}
    for r in range(0,n_class):
    	lam[r] = {}

    for i in range(0,K):
    	for r in range(0,n_class):
		for k in all_popuV:
			T[i][r][k] = 0.0
			N[i][r][k] = 0.0
    for r in range(0,n_class):
	for k in all_popuV:
		lam[r][k] = 0.0
    #step 2.1
    for k in getPopuV():#all_popuV:
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
					l = getDependentV(k)
					for z in l:
						sum_less += N[i][s][z]
				T[i][r][k] = (1.0/servrates[i][r])*(1.0+sum_less)
				#print "T[",i+1,"][",r+1,"][",k, "] = ", T[i][r][k], "1/u: ", 1.0/servrates[i][r], " sum_less: ", sum_less
    #step 2.2
    	for r in range(0,n_class):
		#TODO: get real e: e = getVisitRatio(routL[r], servrates, M):
		sum2 = 0.0
		for i in range(0,K):
			sum2 += e[r][i]*T[i][r][k]
			#print "e[",i+1,"]",e[i],"T[",i+1,"]","[",r+1,"]","[",k,"]",T[i][r][k],"print sum2 in 2.2:", sum2
		lam[r][k] = k[r]/sum2
		#print "lam",r+1,k, "=", lam[r][k]
		#print the throughput of each class for each queue
	if k == (2,2):
		#for i in range(0,K): it's enough to print the info for one queue
		lam1 = e[0][3]*lam[0][k]
		lam2 = e[1][3]*lam[1][k]
		#print "Throughput for queue ", i, " class 1: ", lam1, "class 2:",lam2, "throughput difference: ", lam1 - lam2
		print lam1, lam2, "throughput difference: ", lam1 - lam2
    #step 2.3
    	for i in range(0,K):
    		for r in range(0,n_class):
			N[i][r][k] = lam[r][k]*T[i][r][k]*e[r][i]
			print "N",i+1,r+1,k, "=", lam[r][k],"*",T[i][r][k],"*",e[r][i], "=",N[i][r][k]
			
    return 0
    #use the routL to generate  the visit ratios
    #from nJobList, get the number of jobs in each queue and number of queues
def getVisitRatios(routL):
    l = []
    for i in routL:
	l.append(getVisitRatio(i))
    return l

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

rout = [[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]]
for x in range(1,11): 
	#print "service rate for the reader local computation: ",x
	servrates = [[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1000.0]] #it should be 6 queues
	mva_multiclass([rout,rout], servrates, [2,2], [1,0,1,0])
	#servrates2 = [[1.0,1.0],[1.0,1.0],[1.0,x],[2.0,2.0]] #it should be 6 queues
	#print "adjusting the service rate of the reader threads:"
	#print "The adjusted service rates: ", servrates2
	#mva_multiclass([rout,rout], servrates2, [2,2], [1,0,1,0])
