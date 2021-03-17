# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:28:30 2021

@author: dingxu
"""

#Period: 0.3702918 ID: ZTFJ000006.67+641227.6 SourceID: 55

import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate

CSV_FILE_PATH = '55.csv' #175
dfdata = pd.read_csv(CSV_FILE_PATH)

hjd = dfdata['HJD']
mag = dfdata['mag']

nphjd = np.array(hjd)
npmag = np.array(mag)

HANG = 287  #-1
npmag1 = npmag[0:HANG]-np.mean(npmag[0:HANG])
npmag2 = npmag[HANG:]-np.mean(npmag[HANG:])

npmag = np.concatenate([npmag1,npmag2],axis=0)
#npmag = np.row_stack((npmag1, npmag2))
P = 0.3702918
phases = foldAt(nphjd, P)
sortIndi = np.argsort(phases)
# ... and, second, rearrange the arrays.
phases = phases[sortIndi]
resultmag = npmag[sortIndi]

listmag = resultmag.tolist()
listmag.extend(listmag)

listphrase = phases.tolist()
listphrase.extend(listphrase+np.max(listphrase)) 

indexmag = listmag.index(max(listmag))


nplistphrase = np.array(listphrase)
nplistphrase = nplistphrase-nplistphrase[indexmag]
nplistmag = np.array(listmag)

phasemag = np.vstack((nplistphrase, nplistmag)) #纵向合并矩阵
phasemag = phasemag.T

phasemag = phasemag[phasemag[:,0]>0]
phasemag = phasemag[phasemag[:,0]<1]

#去除异常点
mendata = np.mean(phasemag[:,1])
stddata = np.std(phasemag[:,1])
sigmamax = mendata+2*stddata
sigmamin = mendata-2*stddata

phasemag = phasemag[phasemag[:,1] > sigmamin]
phasemag = phasemag[phasemag[:,1] < sigmamax]


phrase = phasemag[:,0]
flux = phasemag[:,1]
sx1 = np.linspace(0.01,0.99,100)
func1 = interpolate.UnivariateSpline(phrase, flux, s=0.16)#强制通过所有点0.225
sy1 = func1(sx1)


plt.figure(1)
plt.plot(sx1, sy1,'.', c='r')#对原始数据画散点图

plt.figure(0)
plt.plot(phrase, flux,'.')
plt.plot(sx1, sy1,'.', c='r')#对原始数据画散点图
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)

interdata = np.vstack((sx1,sy1))
np.savetxt('ztf1.txt', interdata.T)

praflux = np.vstack((phrase, flux))
np.savetxt('ztf2.txt', praflux.T)



'''
duanx = nplistphrase[188:780]
duany = nplistmag[188:780]
plt.figure(0)
#plt.plot(phases, resultmag, '.')

plt.plot(duanx, duany, '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


a=np.polyfit(duanx,duany,17)#用2次多项式拟合x，y数组
b=np.poly1d(a)#拟合完之后用这个函数来生成多项式对象
c=b(duanx)#生成多项式对象之后，就是获取x在这个多项式处的值
plt.figure(1)
plt.plot(duanx,duany,'.')#对原始数据画散点图
plt.plot(duanx,c,ls='--',c='red')#对拟合之后的数据，也就是x，c数组画图
plt.legend()
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phase',fontsize=14)
plt.ylabel('mag',fontsize=14)
phrasefluxdata = np.vstack((duanx, c))
np.savetxt('lightcurve.txt', phrasefluxdata.T)
'''