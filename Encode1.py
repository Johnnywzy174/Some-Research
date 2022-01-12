#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:54:13 2021

@author: johnnywang
"""



import numpy as np
from numpy import cumsum, log, polyfit, sqrt, std, subtract
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics
from sklearn import preprocessing
import pywt
from glob import glob

df= pd.read_excel("/Users/johnnywang/Documents/Featured Engineering/currency_pairs.xlsx")



def Return (Data):
        Return_df = (Data-Data.shift(1))/Data.shift(1)
        Return_df=Return_df.fillna(0.00)
        
        return Return_df
    

def Encode (Data):
    Enco=[]
    for i in range(len(Data)):
        if Data[i]<= -0.0025:
            Enco.append(-1)
        elif Data[i] >= 0.0025:
            Enco.append(1)
        else:
            Enco.append(0)
    return Enco

#Return
eurusd_return = np.array(Return(df['EURUSD=X']).iloc[1:])
ret_encode=np.array(Encode(eurusd_return))


# decomposition
wavelet_name = "db30"   
signal=df['EURUSD=X']   
coef,freq=pywt.dwt(signal,wavelet_name) 


coef_1=pd.DataFrame(coef)
freq_1=pd.DataFrame(freq)
dwt_coef_return=np.array(Return(coef_1).iloc[1:])
dwt_coef_encode=Encode(dwt_coef_return)
dwt_freq_return=np.array(Return(freq_1).iloc[1:])
dwt_freq_encode=Encode(dwt_freq_return)
dwt_encode=np.array((dwt_coef_encode,dwt_freq_encode)).T


#STD20
stdv_20 = df["EURUSD=X"].rolling(20).std(ddof=0)
stdv_20_=pd.DataFrame(stdv_20).iloc[19:]
stdv_20_return=np.array(Return(stdv_20_).iloc[1:])
stdv_20_encode=np.array(Encode(stdv_20_return))
    


#Normalized approach
normalized_return = preprocessing.normalize([eurusd_return])
normalized_coef = preprocessing.normalize([coef]).T
normalized_freq = preprocessing.normalize([freq]).T
stdv_20=stdv_20[19:]
normalized_stdv_20=preprocessing.normalize([stdv_20])


#Standalized?

    

    
    
    
    
    