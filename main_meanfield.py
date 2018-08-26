# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:40:17 2014

@author: llz
"""
from __future__ import division
from Functions import *
import time
import numpy as np


time1=time.time()

#=======================type 1====================
pars = default_pars_MF()
Stim_time = np.linspace(pars['t_stim']+0.02,pars['t_stim']+pars['dur'],100)
pars['I_ext'] = 40 # with strong input
result=run_sim_mf(pars)


start_time = time.clock()

# V, u, x, I_rec = init_neurons(N,p)


if __name__ == 'main':
	Sim_Time = np.arange(0, p['t_total'], p['dt'])
	spike_record = dict()
	V = np.zeros(p['N'])
	u = np.zeros(p['N'])
	x = np.zeros(p['N'])
	h = np.zeros(p['N'])
	sp = np.zeros(p['N'])



	ux = np.zeros(len(Sim_Time))

    
	end_time = time.clock()
	plt.figure()
	plot_raster(spike_record)
print 'Running time is %0.5f s' % (end_time-start_time)

