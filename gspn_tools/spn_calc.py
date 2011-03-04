#!/usr/bin/env python

# requires installing argparse package.

from argparse import ArgumentParser,FileType
import os,sys,operator,re,subprocess

def todict(tuplelist):
    d = {}
    for k,v in tuplelist:
        d[k] = v

    return d

def getplacedict(name):
    
    netfile = open(name + '.net')
    places = re.findall('^P[0-9]*', netfile.read(), re.MULTILINE)
    pairs = zip(places, map(lambda x: 'p%d' % x, range(1, len(places) + 1)))
    return todict(pairs)

def gettransdict(name):
    stafile = open(name + '.sta')
    trans = re.findall('^Thru_(?P<trans>T[0-9]*) = (?P<thru>\d+\.\d+)$', stafile.read(), re.MULTILINE)
        
    return todict(map(lambda (x,y): (x, float(y)), trans))
    
def getmeans(name, places):
    res = subprocess.Popen(['showtpd', name], stdout=subprocess.PIPE).communicate()[0]
    d = getplacedict(name)
    means = {}
    for p in places:
        probs = re.findall('^' + d[p] + ' prob\((?P<num>\d+)\)=(?P<prob>\d\.\d+)$', res, re.MULTILINE)
        means[p] = sum(map(lambda (x,y): float(x) * float(y), probs))

    return means

def delay(name, places, trans):
    thru = gettransdict(name)
    means = getmeans(name, places)
    return sum(means.values()) / thru[trans]

def main():
    global options
    parser = ArgumentParser()
    #parser.add_argument('--version', action='version', version="%(prog)s 0.01")
    parser.add_argument("-v", dest="verbose", action="store_true", default=False,
                        help="prints verbose data, in the format: wait time, no. threads, lambda, mu")
    parser.add_argument("-t", "--trans", dest="trans", help="no help")
    parser.add_argument("-p", "--places", dest="places", nargs='*',
                        help="from and to values for the lambda parameter.")
    parser.add_argument("inputfile", help="no help.")
    parser.add_argument("-d", "--debug",
                      action="store_true", dest="debug", default=False,
                      help="print debug information")

    options = parser.parse_args()

    if options.debug:
        print >> sys.stderr, "options:", options

    print delay(options.inputfile, options.places, options.trans)

    sys.exit(0)
       
if __name__ == '__main__':
    main() 
