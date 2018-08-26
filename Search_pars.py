# -*- coding: utf-8 -*-
"""
Created on Wed Sep 03 13:22:19 2014


Search for the parameters set of  two decay types

@author: llz
"""

import numpy as np
import time 

# keep down the parameter set that satisfy delta>0
starttime = time.clock()
ts=0.005
pars=[]
for U in np.arange(0.1,0.6,0.05):
    for tr in np.arange(0.01,1,0.01):
        for tf in np.arange(0.01,1,0.01):
            b=1.0/tr+1.0/tf+1.0/(tr+tr*np.sqrt(tf*U/tr))+np.sqrt(U/tr/tf)
            c=2.0/tr/tf+1.0/tr*np.sqrt(U/tr/tf)+1/tr/ts*1.0/(1+np.sqrt(tf*U/tr))-1/tf/ts
            delta=b**2-4*c
            if delta>100 and c>40:
                pars.append([tr,tf,U])
endtime = time.clock()     
print 'Running time is %0.5f s' % (endtime-starttime)      