# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:14:01 2021

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

def plotphoebe(predata,times):
    incl = predata[0]
    q = predata[1]
    r = predata[2]
    t2t1 = predata[3]
    b = phoebe.default_binary(contact_binary=True)
    #times  = np.linspace(0,1,150)
    times = times
    #b.add_dataset('lc', times=phoebe.linspace(0,1,100), passband= 'Kepler:mean')#compute_phases
    b.add_dataset('lc', times=times)

    b['period@binary'] = 1

    b['incl@binary'] = incl #58.528934
    b['q@binary'] =   q
    b['teff@primary'] = 6500#6500#6500  #6208 
    b['teff@secondary'] = 6500*t2t1#6500*92.307556*0.01#6500*100.08882*0.01 #6087

    b['sma@binary'] = 1#0.05 2.32

    b['requiv@primary'] = r    #0.61845703

    #b.add_dataset('mesh', times=[0.25], dataset='mesh01')

    b.run_compute(irrad_method='none')

    fluxmodel = b['value@fluxes@lc01@model']
    resultflux = -2.5*np.log10(fluxmodel)
    resultflux = resultflux - np.mean(resultflux)
    #plt.figure(0)
    #plt.plot(b['value@times@lc01@model'], resultflux, '.')
    return times,resultflux 
'''
file = 'ztf2.txt' #6677225
#file = 'KIC 2437038.txt'
yuandata = np.loadtxt(file)
times = yuandata[:,0]

times,resultflux = plotphoebe(81.306206,0.30078864,0.50842917,1,times)

plt.figure(0)
plt.plot(times, resultflux, '.')
'''
'''
def plotphoebe(incl,q,r,t2/t1,times):
    
    b = phoebe.default_binary(contact_binary=True)
    #times  = np.linspace(0,1,150)
    times = times
    #b.add_dataset('lc', times=phoebe.linspace(0,1,100), passband= 'Kepler:mean')#compute_phases
    b.add_dataset('lc', times=times)

    b['period@binary'] = 1

    b['incl@binary'] = 81.306206 #58.528934
    b['q@binary'] =   0.30078864
    b['teff@primary'] = 5000#6500#6500  #6208 
    b['teff@secondary'] = 5000*1.0#6500*92.307556*0.01#6500*100.08882*0.01 #6087

    b['sma@binary'] = 1#0.05 2.32

    b['requiv@primary'] = 0.50842917    #0.61845703

    b.add_dataset('mesh', times=[0.25], dataset='mesh01')

    b.run_compute(irrad_method='none')

    fluxmodel = b['value@fluxes@lc01@model']
    resultflux = -2.5*np.log10(fluxmodel)
    resultflux = resultflux - np.mean(resultflux)
    #plt.figure(0)
    #plt.plot(b['value@times@lc01@model'], resultflux, '.')
    return times,resultflux 
'''