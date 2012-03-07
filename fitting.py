
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
        pows = np.power(np.repeat(rho, nths -1), np.arange(2,nths+1))
        pi0 = 1./(1. + rho + np.sum(pows))
        w = 1./mu + pi0*rho**2/(lam*(1. - rho)**2)
        return w - y

    def __residual(params, y, n0, n1, n2, x0, x1, x2):
        a0, a1, a2, b0, b1, b2, c0, c1 = params
        return c1 * n0**2 * n1 * n2 * x0 + c0 * n1**2 * n2**2 * x1 * x2 + a0 * n0 ** 2 * x0 + a1 * n1 ** 2 + a2 * n2 ** 2 + b0 * x0 * n0 + b1 * x1 * n1 + b2 * x2 * n2 - y

    p_opt = leastsq(__residual, np.array([10000000,10000, 10000, 1000000, 10000, 10000, 100000, 10000]), args=(y, x[:,0], x[:,1], x[:,2], x[:, 3], x[:, 4], x[:, 5]), full_output = True)
    
    print 'y = %.3f*n0^2*n1*n2*lam0 + %.3f * n1^2 * n2^2 * lam1 * lam2 + %.3f * n0**2 * lam0 + %.3f * n1^2 + %.3f * n2^2 * lam2 +  %.3f * n0 * lam0 + %.3f * n1 * lam1 + %.3f * n2 * lam2' % tuple(map(float, p_opt[0]))
    return p_opt






def __residual2(mu0, n, lam):
    # mu1, fit so that rate stays the same
    rho = lam/mu
    pows = np.power(np.repeat(rho, nths -1), np.arange(2,nths+1))
    pi0 = 1./(1. + rho + np.sum(pows))
    w = 1./mu + pi0*rho**2/(lam*(1. - rho)**2)
    return w

















