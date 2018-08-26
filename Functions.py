# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:03:27 2014
? function file for 'delay_spike'

@author: llz
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def default_pars():
    z = {}
    # the unit of time is sec
    z['dt'] = 0.001 * 0.05
    z['t_total'] = 5
    z['dur'] = 0.3
    z['t_stim'] = 0.5
    z['t_stim2'] = 4.5
    z['N'] = 1000  # total number of excitatry neurons
    z['taus'] = 0.005
    z['taur'] = 0.25
    z['tauf'] = 1.25
    z['taum'] = 0.2
    z['U'] = 0.2
    z['I_ext'] = 4.9
    z['J'] = 5
    z['I_b'] = 2.0
    z['F'] = 900  # Fano factor of the external stimulus
    z['V_th'] = 5
    return z


def default_pars_MF():
    z = {}
    # parameters used in the mean field model, unit sec，
    z['dt'] = 0.001 * 0.05
    z['t_total'] = 5
    z['dur'] = 0.3
    z['t_stim'] = 0.5  # start time of stimulation
    z['taus'] = 5 * 10**(-3)
    z['taur'] = 250 * 10**(-3)
    z['tauf'] = 1250 * 10**(-3)
    z['U'] = 0.2
    z['I_ext'] = 4.9
    z['J'] = 5
    z['I_b'] = 2.0
    z['alpha'] = 1.0
    z['beta'] = 0
    return z


def init_connection(N, density):
    """
    N_E is the number of exicitatory neurons
    N_I is the number of inhibitory neurons
    density is the density of the sparse connection matrix
    W is the connection matrix
    """
    # mean_weight=p['J0']
    # density = p['Connect_p']

    w = np.zeros([N, N])
    # np.random.seed(seed=11)
    w[np.random.rand(N, N) < density] = 1
    return w


def run_sim_mf(p):
    #
    Sim_Time = np.arange(0, p['t_total'], p['dt'])
    results = dict()

    R = np.zeros(len(Sim_Time))
    u = np.zeros(len(Sim_Time))
    x = np.zeros(len(Sim_Time))
    h = np.zeros(len(Sim_Time))
    for i in xrange(0, len(Sim_Time) - 1):
        t = Sim_Time[i]
        if (t > p['t_stim'] and t < (p['t_stim'] + p['dur'])):
            I_ext = p['I_ext']
        else:
            I_ext = 0

        u[i + 1] = u[i] + p['dt'] * \
            (-u[i] / p['tauf'] + p['U'] * R[i] * (1 - u[i]))

        h[i + 1] = h[i] + p['dt'] * \
            (1.0 / p['taus']) * (-h[i] + I_ext +
                                 p['I_b'] + p['J'] * R[i] * u[i + 1] * x[i])

        R[i + 1] = max(0, p['alpha']*h[i + 1] + p['beta'])

        x[i + 1] = x[i] + p['dt'] * \
            ((1 - x[i]) / p['taur'] - x[i] * u[i + 1] * R[i + 1])

    results['t'] = Sim_Time
    results['R'] = R
    results['u'] = u
    results['x'] = x

    return results


def run_sim_spiking(pars):
    Sim_Time = np.arange(0, pars['t_total'], pars['dt'])
    spike_record = dict()
    V = np.zeros(pars['N'])
    u = np.zeros(pars['N'])
    x = np.zeros(pars['N'])
    h = np.zeros(pars['N'])

    for i in xrange(0, len(Sim_Time) - 1):
        t = Sim_Time[i]
        if (t > pars['t_stim'] and t < (pars['t_stim'] + pars['dur'])):
            I_ext = pars['I_ext'] + \
                np.random.randn(pars['N']) * np.sqrt(pars['I_ext'] * pars['F'])

        else:
            I_ext = pars['I_b'] + \
                np.random.randn(pars['N']) * np.sqrt(pars['I_b'] * pars['F'])

        sp = np.zeros(pars['N'])

        sp[V > pars['V_th']] = 1

        u += pars['dt'] * (-u / pars['tauf'] + pars['U'] * sp * (1 - u))

        h += pars['dt'] * (1.0 / pars['taus']) *\
            (-h + I_ext) + pars['J0'] * pars['W'].dot(sp * u * x)

        V += pars['dt'] * (1.0 / pars['taum']) * (-V + h + I_ext)

        sp = np.zeros(pars['N'])

        sp[V > pars['V_th']] = 1

        store_spikes(spike_record, sp, Sim_Time[i])
        x += x + pars['dt'] * ((1 - x) / pars['taur'] - x * u * sp)

    return spike_record


def init_neurons(N, p):
    volt = np.zeros(N)
    u = np.zeros(N)
    x = np.zeros(N)
    V = np.zeros(N)
    I_rec = np.zeros(N)
    return volt, u, x, V, I_rec


def store_spikes(spike_record, sp, t):
    """
    store the current spikes
    t is current time
    spike_record is the dictionary of lists of spike timing
    spiking is the boolean arrays of length n_neuron, j-th entry being 1 indicating neuron j just fired
    """
    for i in xrange(np.size(sp)):
        if sp[i]:
            spike_record[i].append(t)


def plot_raster(spike_record):
    """
    raster plot
    spike_record is the dictionary of lists of spike timing
    """
    # for i in xrange(len(spike_record)):
    for i in xrange(200):
        for j in xrange(len(spike_record[i])):
            plt.plot(spike_record[i][j], i, 'k.', alpha=0.7)
        plt.ylim([0, 200])
        # xlim([1000,3000])
    # xlim([0,1800])
    plt.xlabel('Time (ms)', fontsize=17)
    plt.ylabel('Neuron index', fontsize=17)
