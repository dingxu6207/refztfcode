# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:16:02 2020

@author: dingxu
"""

from tensorflow.keras.models import load_model
#from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing    
#model = load_model('incl.hdf5')
#model = load_model('all.hdf5')
#model = load_model('alldown.hdf5')
#model = load_model('alldrop.hdf5')
#model = load_model('accall.hdf5')
model = load_model('all12.hdf5')

model.summary()

path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 5439790.txt'
#file = 'ztf1.txt'

data = np.loadtxt(path+file)
#data = np.loadtxt(file)

#data[:,1] = -2.5*np.log10(data[:,1])

#datay = 10**(data[:,1]/(-2.5))
#datay = (datay-np.min(datay))/(np.max(datay)-np.min(datay))
datay = data[:,1]-np.mean(data[:,1])

plt.figure(0)
plt.plot(data[:,0], datay, '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phase',fontsize=14)
plt.ylabel('mag',fontsize=14)


plt.figure(1)
hang = data[:,0]*100
inthang = np.uint(hang)
plt.plot(inthang, datay, '.')

npdata = np.vstack((inthang, datay))

lendata = len(npdata.T)

temp = [0 for i in range(100)]
for i in range(lendata):
    index = np.uint(npdata[0,:][i])
    temp[index] = npdata[1,:][i]

        
plt.figure(2)
listtemp = temp[0:50]
resultlist = list(reversed(listtemp))
#temp = temp[0:50]+resultlist
plt.plot(temp, '.')

nparraydata = np.array(temp)
plt.plot(nparraydata, '.' , c = 'blue')
nparraydata = np.reshape(nparraydata,(1,100))

prenpdata = model.predict(nparraydata)

print(prenpdata)


ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phrase',fontsize=14)
plt.ylabel('mag',fontsize=14)