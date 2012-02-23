
import numpy as np
from scipy.optimize import curve_fit, leastsq


def run_fit():
    machine = 'halvan'
    basedir = '/Users/jonatanlinden/Documents/dedup_meas/' + machine + '/memcont/data'
    
    fname = basedir + "/thru_vs_oh_" + machine + ".dat"

    data = np.genfromtxt(fname, delimiter=' ')

    y = data[:,-1]
    x = data[:,0:-1]

    def __residual2(params, y, n0, n1, n2, x0, x1, x2):
        mu, a, b, c,  = params
        nths = np.array(n0 + n1 + n2, dtype=np.int32)
        lam = a*x0 + b*x1 + c*x2
        rho = lam/mu
        print nths
        pows = np.power(np.repeat(rho, nths -1), np.arange(2,nths+1))
        print pows
        pi0 = 1./(1. + rho + np.sum(pows))
        w = 1./mu + pi0*rho**2/(lam*(1. - rho)**2)
        return w - y

    def __residual(params, y, n0, n1, n2, x0, x1, x2):
        a0, a1, a2, b0, b1, b2, c0, c1 = params
        return c1 * n0**2 * n1**2 * n2**2 + c0 * n0 *n1 *n2 * x0 * x1 * x2 + a0 * n0 * x0 ** 2 + a1 * n1 * x1 ** 2 + a2 * n2 * x2 ** 2 + b0 * x0 * n0 + b1 * x1 * n1 + b2 * x2 * n2 - y

    p_opt = leastsq(__residual, np.array([10000000,10000,10000, 1000000, 10000, 10000, 1000000, 10000]), args=(y, x[:,0], x[:,1], x[:,2], x[:, 3], x[:, 4], x[:, 5]), full_output = True)

    #print 'y = %f*n0*(1-lam0)**2 + %f*n0*(1-lam0)**2 + %f*n2*(1-lam2)^2 + %f * n0 (1-lam0) + %f * n1 * (1-lam1) + %f * n2 * (1-lam2)' % tuple(map(float, p_opt[0]))

    return p_opt




