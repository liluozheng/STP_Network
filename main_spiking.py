# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:40:17 2014

@author: llz
"""
from __future__ import division
from Functions import *
import time
import numpy as np


p = default_pars()
N = p['N'] #total number of neuron
p['J0']= 10   
p['t_total'] = 1.8
p['V_th'] = 5
p['U'] = 0.05

start_time = time.clock()

# V, u, x, I_rec = init_neurons(N,p)

p['W'] = init_connection(N, density = 0.1)

Sim_Time = np.arange(0, p['t_total'], p['dt'])
spike_record = dict()
V = np.zeros(p['N'])
u = np.zeros(p['N'])
x = np.zeros(p['N'])
h = np.zeros(p['N'])
sp = np.zeros(p['N'])
for i in xrange(N):
    spike_record[i] = []


ux = np.zeros(len(Sim_Time))
for i in xrange(0, len(Sim_Time) - 1):
    t = Sim_Time[i]
    
    if (t > p['t_stim'] and t < (p['t_stim'] + p['dur'])):
        I_ext = p['I_ext'] + \
            np.random.randn(p['N']) * np.sqrt(p['F'])
    else:
        I_ext = p['I_b'] + \
            np.random.randn(p['N']) * np.sqrt(1* p['F'])

#    sp = np.zeros(p['N'])
#
#    sp[V > p['V_th']] = 1

    u += p['dt'] * (-u / p['tauf']) + p['U'] * sp * (1 - u)

    h += p['dt'] * (1.0 / p['taus']) *\
        (-h + I_ext) + p['J0'] * p['W'].dot(sp * u * x)

    V += p['dt'] * (1.0 / p['taum']) * (-V + h + I_ext)

    sp = np.zeros(p['N'])

    sp[ V > p['V_th']] = 1
    
    V[V > p['V_th']] = 0 # reset

    store_spikes(spike_record, sp, t)
    
    x += p['dt'] * ((1 - x) / p['taur']) - x * u * sp
    
end_time = time.clock()
plt.figure()
plot_raster(spike_record)
print 'Running time is %0.5f s' % (end_time-start_time)

