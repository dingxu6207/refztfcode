# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 21:20:30 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate

CSV_FILE_PATH = 'data\\10.csv'
dfdata = pd.read_csv(CSV_FILE_PATH)

hjd = dfdata['HJD']
mag = dfdata['mag']

rg = dfdata['band'].value_counts()
lenr = rg['r']

nphjd = np.array(hjd)
npmag = np.array(mag)

#hang = 151
#nphjd = nphjd[0:hang]
#npmag = npmag[0:hang]-np.mean(npmag[0:hang])

hang = rg['g']
nphjd = nphjd[hang:]
npmag = npmag[hang:]-np.mean(npmag[hang:])

P = 0.2557214
phases = foldAt(nphjd, P)
sortIndi = np.argsort(phases)
phases = phases[sortIndi]
resultmag = npmag[sortIndi]

plt.figure(5)
plt.plot(phases, resultmag,'.')
ax1 = plt.gca()
ax1.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax1.invert_yaxis() #y轴反向

listmag = resultmag.tolist()
listmag.extend(listmag)

listphrase = phases.tolist()
#listphrase.extend(listphrase+np.max(listphrase)) 
listphrase.extend(listphrase+np.max(1)) 

dexin = int(1*lenr/2)
indexmag = listmag.index(max(listmag[0:dexin]))

nplistphrase = np.array(listphrase)
nplistphrase = nplistphrase-nplistphrase[indexmag]
nplistmag = np.array(listmag)

#phasemag = np.concatenate([nplistphrase, nplistmag],axis=1)


phasemag = np.vstack((nplistphrase, nplistmag)) #纵向合并矩阵
phasemag = phasemag.T

phasemag = phasemag[phasemag[:,0]>=0]
phasemag = phasemag[phasemag[:,0]<=1]

#去除异常点
mendata = np.mean(phasemag[:,1])
stddata = np.std(phasemag[:,1])
sigmamax = mendata+4*stddata
sigmamin = mendata-4*stddata

phasemag = phasemag[phasemag[:,1] > sigmamin]
phasemag = phasemag[phasemag[:,1] < sigmamax]

phrase = phasemag[:,0]
flux = phasemag[:,1]
plt.figure(6)
plt.plot(phrase, flux,'.')




s = np.diff(flux,2).std()/np.sqrt(7.5)
sx1 = np.linspace(0,1,100)
func1 = interpolate.UnivariateSpline(phrase, flux,s=s*s*lenr,ext=3)#强制通过所有点
sy1 = func1(sx1)


plt.figure(1)
plt.plot(sx1, sy1,'.', c='r')#对原始数据画散点图

plt.figure(0)
plt.plot(phrase, flux,'.')
plt.plot(sx1, sy1,'.', c='r')#对原始数据画散点图
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phase',fontsize=14)
plt.ylabel('mag',fontsize=14)

interdata = np.vstack((sx1,sy1))
np.savetxt('ztf1.txt', interdata.T)
praflux = np.vstack((phrase, flux))
np.savetxt('ztf2.txt', praflux.T)

