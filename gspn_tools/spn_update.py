#!/usr/bin/env python

# requires installing argparse package.

from argparse import ArgumentParser,FileType
import os,sys,operator,re

def updatenetfilerate(contents, parameter, value):
    update = re.sub(parameter + r'(\s+) \d+\.\d+e\+\d+ (.*)', 
                    parameter + r'\1' + (' %e' % value) + r' \2',
                    contents)
    return update

def updatenetfiletoken(contents, parameter, value):
    update = re.sub(parameter + r'(\s+) \d+ (.*)', 
                    parameter + r'\1' + (' %d' % value) + r' \2',
                    contents)
    return update

def main():
    global options
    parser = ArgumentParser()
    #parser.add_argument('--version', action='version', version="%(prog)s 0.01")
    parser.add_argument("-v", dest="verbose", action="store_true", default=False,
                        help="no help")
    parser.add_argument("--name", dest="name", help="name of parameter to change in net-file.")
    parser.add_argument("--val", dest="val", type=int, help="the new value to set.")
    parser.add_argument("inputfile", help="no help.")
    parser.add_argument("-d", "--debug",
                      action="store_true", dest="debug", default=False,
                      help="print debug information")

    options = parser.parse_args()

    if options.debug:
        print >> sys.stderr, "options:", options

    netfile  = open(options.inputfile + '.net')
    contents = netfile.read()

    if 0 < len(re.findall(options.name + r'(\s+) \d+\.\d+e\+\d+ (.*)', contents, re.MULTILINE)):
        print updatenetfilerate(contents, options.name, options.val)
    else:
        print updatenetfiletoken(contents, options.name, options.val)
    
    #tmp      = open(name + '~', 'w')
    sys.exit(0)
       
if __name__ == '__main__':
    main() 
