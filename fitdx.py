# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:46:59 2021

@author: jkf
"""
import scipy.signal as ss 
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate
import os,pickle,time
from tensorflow.keras.models import load_model
#import tools_32 as sim
#from pyhht.emd import EMD
import pandas as pd

inclmodel = load_model('incl.hdf5')
allmodel = load_model('all11.hdf5')
dropmodel = load_model('alldrop.hdf5')
accmodel = load_model('accall.hdf5')
alll3model = load_model('l3/alll3.hdf5')
l3model = load_model('l3/l300.hdf5')

def predictdata(sy1):
       
    nparraydata = np.reshape(sy1,(1,100))
    prenpdata = allmodel.predict(nparraydata)
    #print(prenpdata)
    prenpdata[0][1] = prenpdata[0][1]/10
    prenpdata[0][2] = prenpdata[0][2]/100
    prenpdata[0][3] = prenpdata[0][3]/100

    if (prenpdata[0][0]>50) and (prenpdata[0][1]<1.1):     
        prenpdata = dropmodel.predict(nparraydata)
        prenpdata[0][1] = prenpdata[0][1]/100
        prenpdata[0][2] = prenpdata[0][2]/100
        prenpdata[0][3] = prenpdata[0][3]/100

        if (prenpdata[0][0]>70) and (prenpdata[0][1]<0.4):
            prenpdata = accmodel.predict(nparraydata)
            prenpdata[0][1] = prenpdata[0][1]/100
            prenpdata[0][2] = prenpdata[0][2]/100
            prenpdata[0][3] = prenpdata[0][3]/100
     
    print('nol3 = ', prenpdata)
    incldegree = inclmodel.predict(nparraydata)
    print('incl = ', incldegree[0][0])
    return prenpdata,incldegree[0][0]
   
def predictdatal3(sy1): 
    
    #model = load_model('l3/alll3.hdf5')
    nparraydata = np.reshape(sy1,(1,100))

    prenpdata = alll3model.predict(nparraydata)
    #print(prenpdata)
    prenpdata[0][1] = prenpdata[0][1]/10
    prenpdata[0][2] = prenpdata[0][2]/100
    prenpdata[0][3] = prenpdata[0][3]/100
    prenpdata[0][4] = prenpdata[0][4]/100

    if (prenpdata[0][0]>50) and (prenpdata[0][1]<0.8):
        #model = load_model('l3/l300.hdf5')
        prenpdata = l3model.predict(nparraydata)
        prenpdata[0][1] = prenpdata[0][1]/100
        prenpdata[0][2] = prenpdata[0][2]/100
        prenpdata[0][3] = prenpdata[0][3]/100
        prenpdata[0][4] = prenpdata[0][4]/100
        
    print('l3 = ', prenpdata) 
    incldegree = inclmodel.predict(nparraydata)
    return prenpdata,incldegree[0][0]
 
infotemp = []
l3infotemp = []
file='alldata/0000.pkl'
dat=pickle.load(open(file,'rb'))
tot=len(dat)
for i in range(0,tot):
    try:
        ID = dat[i][0]
        name=dat[i][1]
        xy=dat[i][4] 
        num=xy.shape[0]
        

        phrase,flux=xy[:,0],xy[:,1]
        

        sx1 = np.linspace(0,1,100)

        s=np.diff(flux,2).std()/np.sqrt(6)
        print(ID,s)
 
        func1 = interpolate.UnivariateSpline(phrase, flux,k=3,s=s*s*num,ext=3)#强制通过所有点0.225
        sy1 = func1(sx1)
        
        predata,incdata = predictdata(sy1)
        predata = predata[0].tolist()
        #predictdatal3(sy1)
        predata.append(incdata)
        predata.append(ID)
        infotemp.append(predata)
        
        l3predata,l3incdata = predictdatal3(sy1)
        l3predata = l3predata[0].tolist()
        #predictdatal3(sy1)
        l3predata.append(l3incdata)
        l3predata.append(ID)
        l3infotemp.append(l3predata)
        
        plt.clf()
        plt.figure(1)
        plt.plot(phrase,flux,'.')
        plt.plot(sx1,sy1,'.')
        plt.title(ID)
        ax = plt.gca()
        ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
        ax.invert_yaxis() #y轴反向
        #plt.pause(0.5)
        
        
    except:
        print('it is error!'+str(i))
  
name=['incl1','q','r','t2/t1','inc2','ID']      
test = pd.DataFrame(columns=name,data=infotemp)#数据有三列，列名分别为one,two,three
test.to_csv('e:/testcsv.csv',encoding='gbk')

l3name=['incl1','q','r','t2/t1','l3','inc2','ID']      
l3test = pd.DataFrame(columns=l3name,data=l3infotemp)#数据有三列，列名分别为one,two,three
l3test.to_csv('e:/l3testcsv.csv',encoding='gbk')
