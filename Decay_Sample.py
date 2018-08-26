# -*- coding: utf-8 -*-
"""
Created on Fri Aug 08 15:03:14 2014
3 samples of decay behavior
@author: llz

"""
import numpy as np
from Functions import *
import time as time
import matplotlib.pyplot as plt

time1=time.time()

#=======================type 1====================
pars = default_pars_MF()
Stim_time = np.linspace(pars['t_stim']+0.02,pars['t_stim']+pars['dur'],100)
pars['I_ext'] = 40 # with strong input
result=run_sim_mf(pars)



if __name__ == "__main__":
    result  = run_sim_mf(pars)
    plt.figure()
    plt.subplot(3,1,1)
    plt.title('type 1')
    plt.plot(result['t'],result['R'],linewidth=2.5)
    plt.plot(Stim_time,np.zeros(np.size(Stim_time)),'r',linewidth=10)
    plt.xlim([0,3.5])
    plt.ylabel('R',fontsize=13)
#===================type 2=======================
    pars['tauf']=0.8
    pars['taur']=0.01
    pars['U']=0.5
    pars['J']=1.3
    result=run_sim_mf(pars)

    plt.subplot(3,1,2)
    plt.plot(result['t'],result['R'],linewidth=2.5)
    plt.plot(Stim_time,np.zeros(np.size(Stim_time)),'r',linewidth=10)
    plt.xlim([0,3.5])
    plt.ylabel('R',fontsize=13)

	#==================type 3===========================
    pars['tauf']=0.8
    pars['taur']=0.1
    pars['U']=0.1
    pars['J']=3.1
    result = run_sim_mf(pars)
	#subplot(3,1,3)
	#figure()
    plt.subplot(3,1,3)
    plt.plot(result['t'],result['R'],linewidth=2.5)
    plt.plot(Stim_time,np.zeros(np.size(Stim_time)),'r',linewidth=10)
    plt.xlim([0,3.5])
    plt.ylabel('R',fontsize=13)

    plt.xlabel('t(s)',fontsize=14)
    plt.ylabel('R',fontsize=14)

	#subplot(2,1,2)
	#plot(result['t'],pars['J']*result['u']*result['x'],linewidth=2.5,color='purple')
	#xlim([0,4])
	#xlabel('t (s)',fontsize=14)
	#ylabel('Jux',fontsize=14)
    time2=time.time()
    plt.show()
