#!/usr/bin/env python

from lockgraph import parseDicTuple, multi_analyze
import os
path_fmt = os.getcwd() + os.sep + "%i" + os.sep + "output_2_pe_30"
res = {}
fairplot = {}
#for i in [500, 750, 1000, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1500, 1750, 2000, 10000, 50000, 100000]:

#for i in 1400 1500 1600 20000; do
for i in [9500, 40000, 60000, 80000, 90000]:
    path = path_fmt % i
    (tr, ac, re, na) = parseDicTuple(path)
    res[i] = multi_analyze(tr, ac, re, na, [[0], [1]])
    fairplot[i] = (res[i][4][0] - 1./res[i][1][1,0])/(res[i][4][1] - 1./res[i][1][1,1])

for k,v in fairplot.iteritems():
    print "%d\t%.5f" % (k, v)


