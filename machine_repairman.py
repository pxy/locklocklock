#!/usr/bin/env python

# requires installing argparse package.

from argparse import ArgumentParser,FileType
import os,sys,operator

def factorial(n):
    """The factorial function. Included to make it work on old python versions
    """
    return reduce(operator.mul,xrange(2,n+1),1)

def hm(rho,m):
    """Return a function that will take one argument k <= m + 1
    """
    def hm2(k):
        return rho**(k-1)*factorial(m)/factorial(m-k+1)

    return hm2

def mach_rep(lam_lo=1000, lam_hi=1000, mu_lo=1000, mu_hi=1000, m_lo=8, m_hi=8,step=100):
    """does the computation of the waiting times according to the machine repairman queueing model.
    Varies from lambda_low to lambda_hi etc.
    """
    for m in range(m_lo, m_hi+1):
        for i in range(lam_lo, lam_hi+1, step):
            for j in range(mu_lo, mu_hi+1, step):
                lp_lam = float(i)
                lp_mu  = float(j)
                rho = lp_lam/lp_mu
                xs = map(hm(rho,m),range(1,m+2))
                pi0 = 1.0/sum(xs)
                response = float(m)/(lp_mu*(1.0-pi0)) - 1.0/lp_lam
                if options.verbose :print response, m, lp_lam, lp_mu
                else: print response, m

def mach_rep_d(lam_lo=1000, lam_hi=1000, mu_lo=1000, mu_hi=1000, m_lo=8, m_hi=8,step=100):
	"""does the computation of the waiting times according to the machine repairman queueing model with deterministic service time and arrival rate. Varies from lambda_low to lambda_hi etc.
	"""
    	for m in range(m_lo, m_hi+1):
        	for i in range(lam_lo, lam_hi+1, step):
            		for j in range(mu_lo, mu_hi+1, step):
				lp_lam = float(i)
				lp_mu = float(j)
				eo = 1/lp_lam
				ws = 1/lp_mu
				z = eo/ws
				if (m-1)*lp_lam < lp_mu: 
					response = ws
				else: 
					rho = 1
					response = m/lp_mu - eo
                		if options.verbose :print response, m, lp_lam, lp_mu
                		else: print response, m
  
def main():
    global options
    parser = ArgumentParser()
    #parser.add_argument('--version', action='version', version="%(prog)s 0.01")
    parser.add_argument("-v", dest="verbose", action="store_true", default=False,
                        help="prints verbose data, in the format: wait time, no. threads, lambda, mu")

    parser.add_argument("--steps",          dest="steps", type=int, nargs='?',
                        help="no. of steps in total.")
    parser.add_argument("-s", "--stepsize", dest="stepsize", type=int, nargs='?', default=1000,
                        help="the stride size for both lambda and mu values. Default: 1000.")
    parser.add_argument("-m", "--mu"      , dest="ms"      , type=int, nargs=2  , default=[1000,1000],
                        help="from and to values for the mu parameter.")
    parser.add_argument("-l", "--lambda"  , dest="ls"      , type=int, nargs=2  , default=[1000,1000],
                        help="from and to values for the lambda parameter.")
    parser.add_argument("-t", "--threads" , dest="ts"      , type=int, nargs=2  , default=[8,8],
                        help="from and to values for the no. of threads.")
    parser.add_argument("-d", "--debug",
                      action="store_true", dest="debug", default=False,
                      help="print debug information")
    parser.add_argument("--deterministic",
                      action="store_true", dest="deterministic", default=False,
                      help="Set the service time and arrival rate to be deterministic")

    options = parser.parse_args()

    if options.debug:
        print >> sys.stderr, "options:", options

    # maybe not optimal, will not necessarily generate values for the borders of the range.
    if options.steps:
	if options.ls and options.ls[1] > options.ls[0]:
	    options.stepsize = (options.ls[1] - options.ls[0])/(options.steps - 1)
	elif options.ms and options.ms[1] > options.ms[0]:
	    options.stepsize = (options.ms[1] - options.ms[0])/(options.steps - 1)
	    
    if options.stepsize == 0: options.stepsize = 1

    
    if options.deterministic:
    	mach_rep_d(m_lo=options.ts[0], m_hi=options.ts[1], step=options.stepsize,
             lam_lo=options.ls[0], lam_hi=options.ls[1],
             mu_lo=options.ms[0], mu_hi=options.ms[1])
    else:
    	mach_rep(m_lo=options.ts[0], m_hi=options.ts[1], step=options.stepsize,
             lam_lo=options.ls[0], lam_hi=options.ls[1],
             mu_lo=options.ms[0], mu_hi=options.ms[1])
            

    sys.exit(0)
       
if __name__ == '__main__':
    main() 
