#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:32:39 2021

@author: johnnywang
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime as dt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import SpectralClustering
from statistics import mean

import pywt
import yfinance as yf
from hurst import compute_Hc

import warnings
warnings.filterwarnings("ignore")

#Download the data from yahoo Finance
startDate_ = "2010-02-23"
endDate_ = "2021-01-01"

data = yf.download('EURUSD=X', start=startDate_, end=endDate_)
adj_data = data[["Adj Close"]]
adj_data.columns = ['EURUSD=X']

#Calculate the Return and the STD20
def Return (Data):
        Return_df = (Data-Data.shift(1))/Data.shift(1)
        Return_df=Return_df.fillna(0.00)
        
        return Return_df
    
eurusd_return = Return(adj_data['EURUSD=X'])

stdv_5 = pd.DataFrame(adj_data["EURUSD=X"].rolling(5).std(ddof=0))
stdv_10 = pd.DataFrame(adj_data["EURUSD=X"].rolling(10).std(ddof=0))
stdv_20 = pd.DataFrame(adj_data["EURUSD=X"].rolling(20).std(ddof=0))

all_data=pd.concat([adj_data, eurusd_return,stdv_5,stdv_10,stdv_20], axis = 1)
all_data=all_data.iloc[19:]
all_data=all_data.reset_index()
all_data=all_data.reset_index()
all_data.columns=['Day','Date','EURUSD=X','EURUSD_Return','EURUSD_STDEV5','EURUSD_STDEV10','EURUSD_STDEV20']

abs_eurusd_ret= abs(all_data['EURUSD_Return'])
abs_eurusd_ret=pd.DataFrame(abs_eurusd_ret)
abs_eurusd_ret.columns = ['ABS_EURUSD_Return']
all_data = pd.concat([all_data, abs_eurusd_ret], axis=1)

data_len=len(all_data)
clean_len=(data_len//130)*130

#initialize the three groups
df=pd.DataFrame()
non_profit=pd.DataFrame()
outliers=pd.DataFrame()

#for every 130 trading days, we select 10 non-profit(near 0) annd 3 outliers(abs large)
for i in range(0,clean_len,130):
    sorted_1=all_data.iloc[i:i+130]
    sorted_2=sorted_1.sort_values(by='ABS_EURUSD_Return')
    non_profit=non_profit.append(sorted_2.iloc[0:10])
    outliers=outliers.append(sorted_2.iloc[-3:])
    sorted_3=sorted_2.iloc[10:-3]
    sorted_4=sorted_3.sort_values(by='Day')
    df=df.append(sorted_4)
    
df=df.append(all_data.iloc[clean_len:])

# Use Spectral Clustering to cluster the main group
df1=pd.DataFrame()
df1=pd.concat([df['EURUSD_Return'],df['EURUSD_STDEV20']], axis = 1)
clusterer = SpectralClustering(n_clusters = 4, random_state=42)
clusterer.fit(df1)
all_label = clusterer.labels_

#put together all three groups
df['Clustering']=all_label
non_profit['Clustering']=4
outliers['Clustering']=5
final=df.append(non_profit)
final=final.append(outliers)
final=final.sort_index(ascending=True)


cmap = ListedColormap(['r', 'g', 'b','yellow','orange','black'])
norm = BoundaryNorm(range(6), cmap.N)
points = np.array([final.index, final['EURUSD=X']]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(final['Clustering'])
   
fig1 = plt.figure()
plt.gca().add_collection(lc)
plt.xlim(-50, 5000)
plt.ylim(1, 1.62)
plt.show()

final_check=final.copy()

##################Brief study of cluster ##############

########### matrix ###########
coordinate = []
for i in range (0,len(final_check['Clustering'])-1):
    coordinate.append((final_check['Clustering'].iloc[i], final_check['Clustering'].iloc[i+1]))
# determine the size of the matrix
x, y = map(max, zip(*coordinate))

matrix_cnt = np.zeros((x+1, y+1), dtype=float)
matrix = np.zeros((x+1, y+1), dtype=float)
row_sum = np.zeros(x+1,dtype=float)
#count the number of (x,y) at the x,y index
for i, j in coordinate:
    matrix_cnt[i,j] += 1
for i in range(x+1):
    row_sum[i]=sum(matrix_cnt[i,:])
    for j in range(y+1):
        matrix[i,j]=matrix_cnt[i,j]/row_sum[i]
        
        
########## Return #############
Clu0_Ret=[]
Clu1_Ret=[]
Clu2_Ret=[]
Clu3_Ret=[]
Clu4_Ret=[]
Clu5_Ret=[]
for i in range (0,len(final_check['Clustering'])-1):
    if final_check['Clustering'].iloc[i]==0:
        Clu0_Ret.append(final_check['EURUSD_Return'].iloc[i+1])
    elif final_check['Clustering'].iloc[i]==1:
        Clu1_Ret.append(final_check['EURUSD_Return'].iloc[i+1])
    elif final_check['Clustering'].iloc[i]==2:
        Clu2_Ret.append(final_check['EURUSD_Return'].iloc[i+1])
    elif final_check['Clustering'].iloc[i]==3:
        Clu3_Ret.append(final_check['EURUSD_Return'].iloc[i+1])
    elif final_check['Clustering'].iloc[i]==4:
        Clu4_Ret.append(final_check['EURUSD_Return'].iloc[i+1])
    elif final_check['Clustering'].iloc[i]==5:
        Clu5_Ret.append(final_check['EURUSD_Return'].iloc[i+1])

Clu0_Ret_1=[i+1 for i in Clu0_Ret]
Clu1_Ret_1=[i+1 for i in Clu1_Ret]
Clu2_Ret_1=[i+1 for i in Clu2_Ret]
Clu3_Ret_1=[i+1 for i in Clu3_Ret]
Clu4_Ret_1=[i+1 for i in Clu4_Ret]
Clu5_Ret_1=[i+1 for i in Clu5_Ret]



Clustering_data = {'Cluster':['0', '1', '2', '3','4','5'],
        'Avg_Ret':[mean(Clu0_Ret),mean(Clu1_Ret),mean(Clu2_Ret),mean(Clu3_Ret),mean(Clu4_Ret),mean(Clu5_Ret)],
        'Prod_Ret':[np.prod(Clu0_Ret_1),np.prod(Clu1_Ret_1),np.prod(Clu2_Ret_1),np.prod(Clu3_Ret_1),
                   np.prod(Clu4_Ret_1),np.prod(Clu5_Ret_1)],}
Clustering_data=pd.DataFrame(Clustering_data)



############# Trading Strategy Development ############



previous=final_check.iloc[-131:]

#final_check=final.copy()
long_cluster=[0]
short_cluster=[2]
previous['Signal']=0


#final_check=final_check.iloc[-132:]

for i in range (0,len(previous['Clustering'])-1):
    if previous['Clustering'].iloc[i] in long_cluster:
        previous['Signal'].iloc[i+1]=1
    
for i in range (0,len(previous['Clustering'])-1):
    if previous['Clustering'].iloc[i] in short_cluster:
        previous['Signal'].iloc[i+1]=-1
previous=previous.iloc[-130:]
act_ret=[]

for i in range (0,len(previous['Signal'])):
    if previous['Signal'].iloc[i]==1:
        act_ret.append(previous['EURUSD_Return'].iloc[i])
    elif previous['Signal'].iloc[i]==-1:
        act_ret.append(-1*(previous['EURUSD_Return'].iloc[i]))

act_ret_1=[i+1 for i in act_ret]
100*np.prod(act_ret_1)



#final_check1=final_check.copy()

#final_check2=final_check.iloc[:-130]
#a=matrix_cnt_prob(final_check1['Clustering'],final_check2['Clustering'])

#previous.to_excel(r'/Users/johnnywang/Documents/Capstone/Trading/18.xlsx', index = False)



        
#all_return=[i+1 for i in eurusd_return]
#100*np.prod(all_return)