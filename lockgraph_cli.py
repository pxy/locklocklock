#!/usr/bin/env python

# requires installing argparse package.
from argparse import ArgumentParser,FileType
#from lockgraph import analyze, parseDicTuple, servTime
from lockgraph import *
import sys


def main():
    global options
    parser = ArgumentParser()
    parser.add_argument("-v",
                        dest="verbose",
                        action="store_true",
                        default=False,
                        help="print more information.")
    parser.add_argument("-d", "--debug",
                        dest="debug",
                        action="store_true",
                        default=False,
                        help="print debug information")
        
    parser.add_argument("-n",
                        type=int,
                        dest="numCores",
                        nargs='?',
                        help="Number of customers in queueing network.")
    parser.add_argument("datafile",
                        help="Prefix of files containing thread IDs, lock IDs and acquire timestamps, sorted by time. Name should be \"prefix__{try,acq,rel}.dat\"")
    
    options = parser.parse_args()

    if options.debug:
        print >> sys.stderr, "options:", options
        
    lt = ParsedLockTrace(options.datafile)
    lt.analyze()
    lt.mva()

    
    print lt.serv_times()
    print lt.est_wait
    print lt.meas_wait
    
    return 0


if __name__ == '__main__':
	status = main() 
	sys.exit(status)
