#!/usr/bin/env python
import os,sys,math
import numpy as np
from numpy.linalg import solve
np.set_printoptions(threshold=np.nan)

from mva import mva, mva_multiclass


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






    
