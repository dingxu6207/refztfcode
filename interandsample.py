# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:38:42 2021

@author: dingxu
"""

from tensorflow.keras.models import load_model
from scipy import interpolate
#from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing    
#model = load_model('incl.hdf5')
#model = load_model('alldrop.hdf5')
#model = load_model('allsample.hdf5')
#model = load_model('accall.hdf5')
model = load_model('all11.hdf5')

inclmodel = load_model('incl.hdf5')
model.summary()

#path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
#file = 'KIC 9453192.txt'


file = 'ztf1.txt'
data = np.loadtxt(file)
#data[:,1] = -2.5*np.log10(data[:,1])
#data[:,1] = -2.5*np.log10(data[:,1])
#data = np.loadtxt(path+file)
datay = data[:,1]-np.mean(data[:,1])


plt.figure(0)
plt.plot(data[:,0], datay, '.')
plt.xlabel('Phase',fontsize=14)
plt.ylabel('mag',fontsize=14)



sx1 = np.linspace(0,1,100)
func1 = interpolate.UnivariateSpline(data[:,0], datay,s=0.0002)#强制通过所有点
sy1 = func1(sx1)
plt.figure(0)
plt.plot(sx1, sy1, '.', c='r')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phase',fontsize=14)
plt.ylabel('mag',fontsize=14)


nparraydata = np.reshape(sy1,(1,100))

prenpdata = model.predict(nparraydata)
#print(prenpdata)
prenpdata[0][1] = prenpdata[0][1]/10
prenpdata[0][2] = prenpdata[0][2]/100
prenpdata[0][3] = prenpdata[0][3]/100

if (prenpdata[0][0]>50) and (prenpdata[0][1]<1.1):
    model = load_model('alldrop.hdf5')
    prenpdata = model.predict(nparraydata)
    prenpdata[0][1] = prenpdata[0][1]/100
    prenpdata[0][2] = prenpdata[0][2]/100
    prenpdata[0][3] = prenpdata[0][3]/100

    if (prenpdata[0][0]>70) and (prenpdata[0][1]<0.4):
        model = load_model('accall.hdf5')
        prenpdata = model.predict(nparraydata)
        prenpdata[0][1] = prenpdata[0][1]/100
        prenpdata[0][2] = prenpdata[0][2]/100
        prenpdata[0][3] = prenpdata[0][3]/100
     
print(prenpdata)

incldegree = inclmodel.predict(nparraydata)
print('incl = ', incldegree[0][0])

