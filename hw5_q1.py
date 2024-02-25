#! /Users/puyang/anaconda/bin/python 

import numpy as np
from matplotlib import pyplot as plt

def ctmc_simulate(x_0, P, rates, T):
    """
    This function simulates one realization of a Markov chain.
    The states of the Markov chain is denoted as 1, 2, 3...
    Input:
    x_0: intial state of the chain
    P: the transition matrix, a 2-d array or a 2-d numpy array 
    N: the length of the chain to simulate

    Output:
    x: 1-d array where x[i] is the state of the chain at time i
    t: 1-d array where t[i] is the changing time from state x[i-1]
       to state x[i]
    """

    n_state = len(P)
    assert len(rates) == n_state # check we have rates for all states
    x = [x_0];
    t = [0]

    # Unifrom random numbers for simulation
    i = 0
    while t[i] <= T:
        i += 1
        # Simulate x[i], t[i] given x[i-1]
        psum = 0
        Y = np.random.uniform(size=1)
        for j in range(n_state):
            if Y >= psum and Y < psum + P[x[i-1]-1][j]:
                x.append(j+1) # the states of the chain are numbered as 1,2,...
                           # instead of 0,1,...
                t.append(t[i-1] + np.random.exponential(1/rates[x[i-1] - 1]))
                break
            psum += P[x[i-1]-1][j]
    return x, t

def compute_average_performance(x, t, perf):
    """
    This function simulates the total discounted performance of a Markov chain.
    Specifically, it returns a scalar r
        r = sum_{k=0}^N beta^k*perf(x_k),
    where x_k is a sample path of the Marcov chain with transition P given x_0
    Input:
    x: the array of states of the chain
    t: the array of changing times of the chain
    perf: the performance function
    Output:
    r: the simulated total discounted performance
    """
    r = sum([perf(x[i]) * (t[i+1] - t[i]) for i in range(len(x)-2)])
    r += perf(x[len(x)-2]) * (T - t[len(x)-2])
    r /= T
    return r


def get_rate_matrix(P, rates):
    """
    Given the transition matrix and holding rates at each state of a
    CTMC, computes its rate matrix.
    """
    Q = np.zeros((len(P), len(P)))
    for i in range(len(P)):
        for j in range(len(P)):
            if i == j:
                Q[i][j] = -rates[i]
            else:
                Q[i][j] = rates[i]*P[i][j]
    return Q


def compute_stationary_distribution(Q):
    A = np.transpose(Q)
    A[0] = np.ones(len(A), dtype = float)
    pi = np.linalg.solve(A, np.append(1.0, np.zeros(len(A)-1, dtype = float)))
    return pi


def compute_long_run_average(P, rates, perf):
    """
    Compute the long run average performance of a CTMC given its transition matrix 
    and holding rates, by first computing the stationary distirbution of the chain 
    and then compute the expected performance under the stationary distribution.
    """
    Q = get_rate_matrix(P, rates)
    pi = compute_stationary_distribution(Q)
    print (pi)
    res = 0.0
    for i in range(1, len(pi)+1):
        res += perf(i)*pi[i-1]
    return res


if __name__=='__main__':
    x_0 = 1
    
    # Part (a), simulate 1 realization with horizon T=1000
    T = 1000.0
    P = [[0, 0.6, 0.4], [0.3, 0, 0.7], [0.85, 0.15, 0]] 
    rates = [0.1, 0.2, 0.3]
    x, t = ctmc_simulate(x_0, P, rates, T)
    plt.scatter(t, x, s=1)
    plt.xlim(0, 1000)
    plt.ylim(0, 4)
    plt.xlabel('$t$')
    plt.ylabel('state')
    plt.savefig('q1_a.pdf', bbox_inches='tight')

    # Part (b), simulate 1 realization with n=1000 and compute the average
    # performance of this realization
    # compute long run average performance of this chain by first computing the
    # stationary distribution
    perf = lambda x: x*x
    avg = compute_average_performance(x, t, perf)
    avg_theoretical = compute_long_run_average(P, rates, perf)
    print ("The average performance of this realization is: {}.".format(avg))
    print ("Expected infinite horizon long run average performance is: {}.".format(
        avg_theoretical))
