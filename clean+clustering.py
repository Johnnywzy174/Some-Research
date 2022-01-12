#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 09:41:15 2021

@author: johnnywang
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime as dt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import SpectralClustering

import pywt
import yfinance as yf
from hurst import compute_Hc

import warnings
warnings.filterwarnings("ignore")

#Download the data from yahoo Finance
startDate_ = "2007-01-01"
endDate_ = "2021-03-01"

data = yf.download('EURUSD=X', start=startDate_, end=endDate_)
data = data[["Adj Close"]]
data.columns = ['EURUSD=X']

#Calculate the Return and the STD20
def Return (Data):
        Return_df = (Data-Data.shift(1))/Data.shift(1)
        Return_df=Return_df.fillna(0.00)
        
        return Return_df
    
eurusd_return = Return(data['EURUSD=X'])
stdv_20 = pd.DataFrame(data["EURUSD=X"].rolling(20).std(ddof=0))
all_data=pd.concat([data, eurusd_return,stdv_20], axis = 1)
all_data=all_data.iloc[19:]
all_data.columns=['EURUSD=X','EURUSD_Return','EURUSD_STDEV20']


#initialize the three groups
df=pd.DataFrame()
non_profit=pd.DataFrame()
outliers=pd.DataFrame()

#for every 125 trading days, we select 10 non-profit(near 0) annd 3 outliers(abs large)
for i in range(0,3631,125):
    sorted_1=all_data.iloc[i:i+125]
    sorted_2=pd.concat([sorted_1, abs(sorted_1['EURUSD_Return'])], axis = 1)
    sorted_2.columns=['EURUSD=X','EURUSD_Return','EURUSD_STDEV20','ABS_EURUSD=X']
    sorted_3=sorted_2.sort_values(by='ABS_EURUSD=X')
    non_profit=non_profit.append(sorted_3.iloc[0:10])
    outliers=outliers.append(sorted_3.iloc[-3:])
    sorted_4=sorted_3.iloc[10:-3]
    sorted_5=sorted_4.sort_index(ascending=True)
    df=df.append(sorted_5)

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


#Plot the time Series Data with the color indicates each clustering 

final1=final.reset_index()

cmap = ListedColormap(['r', 'g', 'b','yellow','orange','black'])
norm = BoundaryNorm(range(6), cmap.N)
points = np.array([final1.index, final1['EURUSD=X']]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(final['Clustering'])
   
fig1 = plt.figure()
plt.gca().add_collection(lc)
plt.xlim(-50, 3700)
plt.ylim(1, 1.62)
plt.show()
    
#final.reset_index().plot(x='Date', y='EURUSD=X')