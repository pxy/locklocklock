#!/usr/bin/env python

import os,sys,math
import operator as op
import numpy as np
from numpy.linalg import solve
from itertools import *


# STANDARD MVA

def st_mva(p, u, M, q_type=None):
    K = p.shape[0] #number of queues

    # we assume every second queue is infinite server (i.e., think time)
    if not q_type:
        q_type = np.array(map (lambda (i): i % 2, range(K)))

    v = getVisitRatio(p)

    N = np.zeros((K,M+1)) # matrix for queue length of node K with M customers 
    W = np.zeros((K,M+1)) # response time of node K with M customers

    for m in range(1,M+1): #[1..M] #loop over the number of customers
        # first column (m) is always zero

        #step 1
        # for each queue, update the waiting time
        W[:,m] = (q_type*N[:,m-1] + 1.0)/u

		#step 2, throughput for the whole network
        lam = m/np.dot (W[:,m],v)

        #step3
        N[:,m] = lam*v*W[:,m]

    # return the waiting times for the case when M customers are in the network
    return W[:,M], N[:,M]


# single class mva with load dependent service rate
# e visit ratios
# ld_mu, load dep servrates, ld_mu[i] := serv rates for node i
# no IS nodes, replace by increasing serv.rates in ld_mu
# POST: Wait time, norm. const. and marg. probs. when M cust. are present in net.
def ld_mva(e, ld_mu, M):
    K = e.shape[0]               #number of queues
    sp = np.zeros((K, M+1, M+1)) # state probs
    T = np.zeros((K,M+1))        # response time of node K with M customers
    G = np.zeros((M+1))          # norm. const.
    sp[:,0,0] = 1.0
    G[0] = 1.0

    # loop over the number of customers
    # [1..M], first column, T[:,0] is always zero
    for m in range(1,M+1):

        # step 1 : for each queue, update the waiting time
        for i in range (K):
            #T[i,m] = sum ([(j/ld_mu[i,j-1]) * sp[i,j-1,m-1] for j in range(1, m+1)])
            T[i,m] = sum ([(j/ld_mu[i]) * sp[i,j-1,m-1] for j in range(1, m+1)])

		#step 2 : total throughput of network
        lam = m/np.dot (T[:,m],e)

        # norm. const.
        G[m] = G[m-1]/lam

        # update state probs
        for i in range (K):
            #sp[i,1:m+1,m] = (lam/ld_mu[i,1:])*sp[i,:m,m-1]*e[i]
            sp[i,1:m+1,m] = (lam/ld_mu[i])*sp[i,:m,m-1]*e[i]
            sp[i,0,m] = 1. - sum (sp[i,:m+1,m])
            
    # return the waiting times for the case when M customers are in the network
    return T[:,M], G[M], sp[:,:,M]



# MULTICLASS MVA

#This function implements the mean value analysis for the multi-class closed network
#    inputs: routL: routing matrix list
#	 servrates: service rates for the queues for different job classes, should be a #queues x # classes matrix
#	 nClassL: a list of the number of threads in each class, the length of the list is the number of classes
def mva_multiclass(routL, servrates, nClassL, queueType, vr=None):
    #total number of queues and classes
    K = len(queueType)
    n_class = len(routL)
    all_popuV = getPopulationVs(nClassL)
    if vr != None:
        e = vr
    else:
        e = np.array(getVisitRatios(routL))

    #STEP 1: initialize the number of jobs matrices,
    # N and T are dictionaries of matrices, the keys are the pop.vectors
    # Rows are the different values for the classes, hence #rows == #queues
    # Columns are the classes, hence index [k][q,c] is the value for queue q and class c,
    # when the population for each class is specified by k
    # lam is throughput
    # NB: e has other interpretation of the dimensions: #rows = #classes
    T = {}
    N = {}
    lam = {} 
    for k in all_popuV:
        T[k] = np.zeros((K,n_class))
        N[k] = np.zeros((K,n_class))
        lam[k] = np.zeros(n_class)

    U = np.zeros((K, n_class))
    # ***BEGIN ALGO***

    for k in all_popuV:
        #STEP 2.1
        # calculate T
        for i in range(K): # queues

            # if node i is an infinite server node, the response
            # time is just the service time
            if queueType[i] == 1:
                T[k][i] =  1.0/servrates[i]

            # if node i is a single server queue
            else:
                # T[k] is total service time for expected no.
                # of cust waiting + new job
                # A_k is the sum of the service times of the jobs waiting at a server at
                # the arrival of a new job
                A_k = np.array([(N[dependentV(k, x)][i]*(1.0/servrates[i])).sum() for x in range(n_class)])
                T[k][i] = (1.0/servrates[i] + A_k) # R_ck
                                
        #STEP 2.2
        # calculate throughput
        # for each class/row, sum together expected time
        sum2 = np.diag(np.dot(e, T[k]))
        lam[k] = np.array(k)/sum2
        
        #STEP 2.3
        # for each class and each server, update est. no. of
        # customers in server.
        N[k] = T[k]*lam[k]*e.T

    # util.
    
    for i in range(K):
        # utilization is relative mean fraction of active servers, hence
        # in the infinite server case utilization is always zero
        if not queueType[i]:
            U[i] = e[:,i]*lam[nClassL]/servrates[i]
    # ***END ALGO*** for loop over pop.vectors.
    return  T[nClassL], N[nClassL], U




# ****************************** MARIE'S METHOD ********************************

# relation 5 in baynat paper
# POST: load-dependent arrival rates
def rel5 (_ld_mu, marg_p):
    # hack
    if type(_ld_mu) is np.float64:
        ld_mu = np.array([_ld_mu])
    else:
        ld_mu = _ld_mu
    ld_lam = np.zeros_like(ld_mu)
    for n in range(ld_lam.shape[0]):
        ld_lam[n] = ld_mu[n]*marg_p[n+1]/marg_p[n]
    return ld_lam

# relation 3 in baynat paper
# POST: conditional throughput, [1, N]
# ld_lam, [0, N-1]
# marg_p, [0, N]
def rel3(ld_lam, marg_p):
    cond_nu = np.zeros_like(ld_lam)
    for n in range(1, cond_nu.shape[0]+1):
        cond_nu[n-1] = ld_lam[n-1]*marg_p[n-1]/marg_p[n]
    return cond_nu


# ld_lam: array with pop. dependent arrival rates
# lam_r : fixed retrial overhead
# mu    : fixed service time
def gen_mc (ld_lam, lam_r, mu):
    n_states = ld_lam.shape[0]*2
    gen_mtx  = np.zeros((n_states, n_states))

    # limit cases
    gen_mtx[0,1] = ld_lam[0]
    gen_mtx[n_states-1, n_states-2] = mu

    # every second row, C(x) = 0, every second row, C(x) = 1
    # starting with state 2
    for st in range(1, n_states - 1):
        nc = int(math.ceil(st/2.0))
        if st % 2:
            gen_mtx[st, st-1] = mu
            gen_mtx[st, st+2] = ld_lam[nc]

        else:
            gen_mtx[st, st-1] = lam_r
            gen_mtx[st, st+1] = ld_lam[nc]

    # fill diagonal
    return gen_mtx + (-gen_mtx.sum(axis=1)) * np.identity(n_states)


# ld_lam : first dim, class, sec dim ncust
def gen_mc2 (ld_lam, lam_r, mu_c):
    n_states = 7
    gen_mtx  = np.zeros((n_states, n_states))

    # limit cases
    gen_mtx[0,1] = ld_lam[0,0]
    gen_mtx[0,2] = ld_lam[1,0]

    # 1: 1000
    gen_mtx[1,0] = mu_c[0]
    gen_mtx[1,3] = ld_lam[1,0]

    # 2: 0100
    gen_mtx[2,0] = mu_c[1]
    gen_mtx[2,4] = ld_lam[0,0]

    # 3: 1001
    gen_mtx[3,5] = mu_c[0]

    # 4: 0110
    gen_mtx[4,6] = mu_c[1]

    # 5: 0001
    gen_mtx[5,3] = ld_lam[0,0]
    gen_mtx[5,2] = lam_r
    
    # 6: 0010
    gen_mtx[6,4] = ld_lam[1,0]
    gen_mtx[6,1] = lam_r

    return gen_mtx + (-gen_mtx.sum(axis=1)) * np.identity(n_states)
                       
def retrial_margp(steadystate):
    n_cust = steadystate.shape[0]/2+1
    marg_p = np.zeros(n_cust)
    marg_p[0] = steadystate[0]
    marg_p[-1] = steadystate[-1]
    for n in range(1, n_cust - 1):
        marg_p[n] = steadystate[2*n - 1] + steadystate[2*n]
    return marg_p

def retrial_margp_c (st):
    margp = np.zeros((2,2))
    margp[0,1] = st[1]+st[3] + st[4] + st[6]
    margp[1,1] = st[2]+st[3] + st[4] + st[5]
    margp[0,0] = 1. - margp[0,1]
    margp[1,0] = 1. - margp[1,1]
    return margp
    
# try2D, acq2D, rel2D, names2D = parseDicTuple('/Users/jonatanlinden/res/micro/output_2_pe_30')
# analres = multi_analyze(try2D, acq2D, rel2D, names2D, [[0],[1]])

# rout : list of routing mtcs
# serv_times : matrix with load dependent serv rates per class
# serv_times[class, q, load]
def run_marie(rout, serv_times):
    #ld_mu_c0[server, load]
    #ld_mu_c0 = np.vstack ((serv_times, serv_times)).T
    #ld_mu_c0[0,1] = 2*ld_mu_c0[0,1]
    #ld_mu_c1 = ld_mu_c0.copy()

    # in the analysis in isolation, the mu doesn't change
    fix_mu = serv_times[:,1]

    ld_mu_c0 = serv_times[0].copy()
    ld_mu_c1 = serv_times[1].copy()
    
    old_ld_mu_c0 = np.zeros_like(ld_mu_c0)
    old_ld_mu_c1 = np.zeros_like(ld_mu_c1)
    e = getVisitRatio(rout[0])
    lam_r = 1/4540.
    while (not (mtx_eq(old_ld_mu_c0, ld_mu_c0, 0.00001*np.max(ld_mu_c0)) and mtx_eq(old_ld_mu_c1, ld_mu_c1, 0.00001*np.max(ld_mu_c1)))):
        _,_,marg_p_c0 = ld_mva(e, ld_mu_c0, 1)
        _,_,marg_p_c1 = ld_mva(e, ld_mu_c1, 1)


        if conv_cond0(marg_p_c0, 0.01):
            print "cond0"
            break
        
        # ld_lam, defined for [0, N-1]
        # ld_lam[server, load]
        ld_lam_c0 = rel5(ld_mu_c0[1], marg_p_c0[1])
        ld_lam_c1 = rel5(ld_mu_c1[1], marg_p_c1[1])
        print np.array([ld_lam_c0, ld_lam_c1])
        #mc = gen_mc(ld_lam_c0, lam_r, serv_times[1])
    
        mc = gen_mc2(np.array([ld_lam_c0,ld_lam_c1]), lam_r, fix_mu)

        steadystate = solve_ctmc(mc)
        loc_marg_p = retrial_margp_c(steadystate)
        #loc_marg_p = retrial_margp(steadystate)
        print loc_marg_p
        cond_thr_c0 = rel3(ld_lam_c0, loc_marg_p[0])
        cond_thr_c1 = rel3(ld_lam_c1, loc_marg_p[1])
        #print np.array([cond_thr_c0, cond_thr_c1])
        old_ld_mu_c0 = ld_mu_c0.copy()
        old_ld_mu_c1 = ld_mu_c1.copy()
        ld_mu_c0[1] = cond_thr_c0
        ld_mu_c1[1] = cond_thr_c1

    print "exiting:"
    print ld_mu_c0
    print old_ld_mu_c0
    print ld_mu_c1
    print old_ld_mu_c1

    print ld_mva(e, ld_mu_c0, 1)
    print ld_mva(e, ld_mu_c1, 1)
    return ld_mu_c0


def conv_cond0(marg_p, eps):
    n_cust = marg_p.shape[1]
    n_s = marg_p.shape[0]
    zero_c = (n_cust - sum ([(sum ([n*marg_p[i,n] for n in range(n_cust)])) for i in range(n_s)]))/n_cust
    return zero_c < eps

# ************************* HELPER FUNCTIONS ************************************

def mtx_eq (mtx0, mtx1, epsilon):
    return np.alltrue(np.absolute(mtx1 - mtx0) < epsilon)

def solve_ctmc(mc):
    '''
    Returns the steady state probs of the markov chain mc
    mc: markov chain of a closed queueing network
    '''
    n_states = mc.shape[0]
    a = np.zeros(n_states+1) # Zero vector with the last element being 1.0
    a[-1] = 1.0
    r = np.vstack((mc.T, np.ones(n_states)))
    return np.linalg.lstsq(r, a)[0]

# return prob of state with pop vector K, K[i] #cust at node i in load-dep
def state_prob (G, K, e, ld_mu):
    n_serv = len (K)
    prod = reduce (op.mul, [F_FCFS_singleserv([K[i]], e, ld_mu[i]) for i in range(n_serv)], 1.0)
    return 1/G*prod

# S class pop vector
# F_i from Gunther Bolsch book
def F_FCFS_singleserv (S, e, ld_mu):
    k = sum (S)
    n_class = len (S)
    k_fac = factorial(k)
    # assumes same service rate for all classes - not a problem in single class setting
    if k > 0:
        prod_0 = reduce (op.mul, [ld_mu[i] for i in range(k)], 1)
    else:
        prod_0 = 1.0
    print prod_0
    prod_1 = reduce (op.mul, [1.0/factorial(S[r]) * e[r]**S[r] for r in range(n_class)],1)
    print prod_1
    return k_fac/prod_0*prod_1


# mean value of retrial queue with deterministic retrial time, only retrial
# of head of the orbit, and new retrial time beginning at a service completion epoch
def mean_retr(detrate, exprate, servrate, e_res):
    rho = exprate/servrate

    alpha = np.exp(-exprate*detrate)
    dirt = rho/(alpha-rho)*(e_res + (1.0-alpha)/exprate)
    
    return np.nan_to_num(dirt)
    

def getVisitRatios(routL):
    return map(getVisitRatio, routL)


def getVisitRatio(p):
    K = len(p) #number of queues
    # A K by K identify matrix to be used later to solve the traffic equation
    I = np.identity(K)
    #substitute the last row in the transpose matrix of I-p with an array with ones
    # and try to solve that equation (normalize the ratios in the traffic equation)
    q = np.ones(K)
    tmp = (I-p).T
    r = np.vstack((tmp[:-1,:],q))

    a = np.zeros(K) # Zero vector with the last element being 1.0
    a[K-1] = 1.0
    v = solve(r,a)
    return v

# all possible class pop. vectors, less than the list nClassL
def getPopulationVs (nClassL):
    return list(product(*[range(i+1) for i in nClassL]))

# POST: the unit vectors in dimension dim as tuples
def unitVs (dim):
    return map (tuple, np.identity(dim, dtype=int))

# POST: the dependent pop vector of tup where class c has one less customer
def dependentV(tup, c):
    unit = unitVs(len(tup))[c]
    res = np.subtract(tup, unit)
    res[res<0] = 0
    return tuple(res)

def factorial(n):
    if not n >= 0:
        raise ValueError("n must be >= 0")
    if math.floor(n) != n:
        raise ValueError("n must be exact integer")
    if n+1 == n:  # catch a value like 1e300
        raise OverflowError("n too large")
    result = 1
    factor = 2
    while factor <= n:
        result *= factor
        factor += 1
    return result

# *************************** TEST CASES *****************************************

# this test should return true :)
def closedsingleclasstest():
    p_back = 0.5
    R = np.zeros((7,7))
    R[0,1] = 1.0
    R[1,2:5] = 1.0/3
    R[2:5,5:7] = (1-p_back)/2.0
    R[2:5,0] = p_back
    R[5:7,2:5] = 1/3.0
    S = np.array([0.2, 2, 1/0.8, 1/0.8, 1/0.8, 1/1.8, 1/1.8])
    correct_answer = np.array([[  5.        ],
                               [  0.97950777],
                               [  1.67066031],
                               [  1.67066031],
                               [  1.67066031],
                               [ 10.63442802],
                               [ 10.63442802]])
    t,n = mva_multiclass([R], S, (20,), [1,0,0,0,0,0,0])
    return np.alltrue(np.absolute(t - correct_answer) < 0.00001)
    

# this one is not verified
def closedmclasstest1():
    S = np.array([[1, 0.25, 0.125, 1/12.0],
                  [0.5, 0.2, 0.1, 1/16.0]])
    e = np.array([[1, 0.4, 0.4, 0.2],
                  [1, 0.4, 0.3, 0.3]])
    nPop = (1,2)
    return mva_multiclass ([], S.T, nPop, [0,0,0,1], vr = e)

def closedmclasstest2():
    S = np.array([[1,1/2.0],[1/3.0,1/4.0]])
    r = np.array([[0, 1], [1, 0]])
    nPop = (1,1)
    correct_answer = np.array([[ 1.33333333,  2.5       ],
                               [ 5.        ,  7.        ]])


    t, n = mva_multiclass ([r, r], S, nPop, [0,0])
    return np.alltrue(np.absolute(t - correct_answer) < 0.00001)


# APPROX ITERATIVE MULTICLASS MVA WITH DETERMINISTIC SERVICE TIMES
# not working
def mva_dsa_multiclass(routL, servrates, nClassL, queueType, vr=None):


    K = len (servrates)
    nC = len (routL)
    U = np.zeros((K, nC))
    e = np.array(getVisitRatios(routL))
    cnt = 0
    # scale factors for each server
    sf = np.ones(len(servrates))
    T, N, Unew = mva_multiclass(routL, servrates, nClassL, queueType, vr, sf)

    while not (np.alltrue(np.absolute(U - Unew) < 0.0001) or cnt > 4):
        U = np.copy(Unew)
        print "U: ", U
        # solve for scale factors

        # sum row wise
        rho_c = U.sum(axis=1)
        print "rho_c: ", rho_c

        sumc = np.zeros((nC,K))
        for c in range (nC):
            uhat = rho_c - U[:,c]
            sumc[c] += (uhat * e[c])

        print "Sumc: ", sumc
        rho = sumc.sum(axis=1)/e.sum(axis=1)
        rho = rho * (nC/(nC-1))
        print "rho: ", rho
        sf = (2.0 - rho) / (2.0 - np.power(rho, 2))
        print "scale factors: ", sf
        T, N, Unew = mva_multiclass(routL, servrates, nClassL, queueType, vr, sf)
        
        cnt += 1

    print "iterations: ", cnt
    return  T, N, Unew
