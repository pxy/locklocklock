#!/usr/bin/env python

# requires installing argparse package.
from argparse import ArgumentParser,FileType
from lockgraph import analyze, parseDicTuple, servTime 
import sys


def main():
	global options
	parser = ArgumentParser()
	parser.add_argument("-v", dest="verbose", action="store_true", default=False,
						help="print more information.")
	parser.add_argument("-d", "--debug",
						action="store_true", dest="debug", default=False,
						help="print debug information")
	parser.add_argument("-n", type=int, dest="numCores", nargs='?', help="Number of customers in queueing network.")
	parser.add_argument("datafile", help="Prefix of files containing thread IDs, lock IDs and acquire timestamps, sorted by time. Name should be \"prefix_{#threads}th_{acq,rel}.dat\"")
	parser.add_argument("servfile", help="File containing lock IDs and average service time, sorted by lock ID.")

	options = parser.parse_args()

	if options.debug:
		print >> sys.stderr, "options:", options

	# parse data for 8 thread case
	(acq8D, rel8D, create8Vec, instr8Vec_) = parseDicTuple (options.lockacq, 8)

	# parse data for 1 thread case
	(acq1D, rel1D, create1Vec, instr1Vec_) = parseDicTuple (options.lockacq, 1)

	(servTimeArr, count) = servTime (acq1D, rel1D)

	# (servD1, mapp) = lockMap (hexToLine(instr8Vec_, '/Users/jonatanlinden/res/dedup'),
	# 						  hexToLine(instr1Vec_, '/Users/jonatanlinden/res/dedup'),
	# 						  servTimeArr, count)

	# add missing entries to servD1 manually, then run the following code
	#mapD
	#for k in mapp.keys():
	#    mapD[k] = servD1[mapp[k]]
	#servTimeVec = dictToArray (mapD)
	
	analyze (acqDic, relDic, servTimeVec, options.numCores)
	return status


if __name__ == '__main__':
	status = main() 
	sys.exit(status)
