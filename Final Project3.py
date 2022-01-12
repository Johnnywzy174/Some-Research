#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:41:34 2020

@author: Johnny Wang
"""

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import ConvLSTM2D
from numpy import hstack
from keras.layers import RepeatVector
import yfinance as yf
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Lasso
from matplotlib import pyplot as plt
"""
Set the range
"""
pd.set_option('display.max_rows', 3888)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
file = '/Users/johnnywang/Documents/Featured Engineering/Currencies_and_SP500(1).csv'
# Set rolling window range
range_period_list = list(range(150,160,10))
# Read file
df = pd.read_csv(file,
                 names=['Date', 'EURUSD=X','S&P','EURUSD_Cluster','S&P_Cluster','STD', 'STD_Cluster','Vol_global', 
                        'Vol_global_Cluster','Vol_local', 'Vol_local_Cluster','Primary risk', 'Primary risk_Cluster',
                        'FD','FD_Cluster','Hurst', 'Hurst_Cluster','Correlation', 'Correlation_Cluster','Energy',
                        'Energy_Cluster','Return', 'Return_Cluster'])
regressor_list = []    

def Clusters (Data):
    
        Data = Data.fillna(0)

        X = Data.values.reshape(-1,1)

# initialize k-means algo object with 4 clusters

        kmeans = KMeans(n_clusters=4, init='k-means++', tol=0.004, max_iter=300, random_state=42).fit(X) #get kmeans

        y_kmeans = kmeans.predict(X)


        return y_kmeans
    
for range_period in range_period_list:
    """
    Calculation
    """
    # range_period = 150 #//the best window in this condition - see lines 115-130: time window optimization
    # calculate the std within range 
    df['STD'] = df['EURUSD=X'].rolling(range_period).std(ddof=0)

    # get the std_max to calculate the vol_global
    df_std_max = df['STD'].max()
    # calculate the vol_global
    df['Vol_global'] = df['STD'] / df_std_max

    # calculate the vol_local
    df['Vol_local'] = df['STD'] / df['STD'].rolling(10).mean() / 3

    # calculate the primary risk using np.ceil (round up)
    df['Primary risk'] = (1 / df['Vol_global']).apply(np.ceil) / 10

    # calculate the correlation
    df['Correlation'] = 0.5 * (df['EURUSD=X'].rolling(range_period).corr(df['S&P']) + 1)

    # Calculate FD and Hurst exponent
    FDs = list()
    Hurst = list()
    for i, data_row in df.iterrows():
        if i < len(df) + 1 - range_period:
            spy_close = df[i:i + range_period - 1]
            spy_close = spy_close[['EURUSD=X']].copy()
            lag1, lag2 = 2, 20  # lag chosen 2 , 20
            lags = range(lag1, lag2)

            # Fractal Dimension calculation
            tau = [sqrt(std(subtract(spy_close[lag:], spy_close[:-lag]))) for lag in lags]
            m = polyfit(log(lags), log(tau), 1)
            hurst = m[0] * 2
            fractal = 1 - hurst[0]
            FDs.append(fractal)
            Hurst.append(hurst)
        else:
            break
    df['FD'][range_period - 1:len(df) + 1] = FDs
    df['Hurst'][range_period - 1:len(df) + 1] = Hurst

    # Linear regression to find coeff for calculating the energy
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    Vols = df['Vol_global'][range_period - 1:len(df) + 1]
    FD = df['FD'][range_period - 1:len(df) + 1]
    X = Vols.values.reshape(-1, 1)
    y = FD.values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Linear Regression model built on train data
    regressor = LinearRegression()

    # Training the algorithm
    regressor.fit(X_train, y_train)

    # Calculating the coeff B of Fractal Dim
    c = regressor.intercept_
    B = 1 / c

    # Calculating the coeff A of Vol
    m = regressor.coef_
    A = m.item() * B
   

    # Calculate the Energy
    df['Energy'] = A * df['Vol_global'] + B * df['FD']
    energy_list = list(df['Energy'].values)
    energy_list = energy_list[range_period-1:]
    response_list = np.array(range(len(energy_list)))
    #fit the linear regression between energy and time
    regressor.fit(np.array(energy_list).reshape(-1,1),response_list.reshape(-1,1))
    regressor_list.append(regressor)
    
    df['Return'] = (df['EURUSD=X'] - df['EURUSD=X'].shift(1)) / df['EURUSD=X'].shift(1)

    df['Return'] = df['Return'].fillna(0)
    

    
    
    df['EURUSD_Cluster']=Clusters(df['EURUSD=X'])
    df['S&P_Cluster']=Clusters(df['S&P'])
    df['STD_Cluster']=Clusters(df['STD'])
    df['Vol_global_Cluster']=Clusters(df['Vol_global'])
    df['Vol_local_Cluster']=Clusters(df['Vol_local'])
    df['Primary risk_Cluster']=Clusters(df['Primary risk'])
    df['FD_Cluster']=Clusters(df['FD'])
    df['Hurst_Cluster']=Clusters(df['Hurst'])
    df['Correlation_Cluster']=Clusters(df['Correlation'])
    df['Energy_Cluster']=Clusters(df['Energy'])
    df['Return_Cluster']=Clusters(df['Return'])

'''    
d = {'EURUSD_Cluster': Clusters(df['EURUSD=X']), 'S&P_Cluster': Clusters(df['S&P']),
     'STD_Cluster': Clusters(df['STD']),'Vol_global_Cluster': Clusters(df['Vol_global']),
     'Vol_local_Cluster': Clusters(df['Vol_local']), 'Primary risk_Cluster':Clusters(df['Primary risk']),
     'FD_Cluster': Clusters(df['FD']), 'Hurst_Cluster': Clusters(df['Hurst']),
     'Correlation_Cluster': Clusters(df['Correlation']), 'Energy_Cluster':Clusters(df['Energy']),
     'Return_Cluster': Clusters(df['Return'])}
df1 = pd.DataFrame(d)
df.to_csv(r'/Users/LONG/Desktop/data.csv',index=False)

df1.to_csv(r'/Users/LONG/Desktop/data1.csv',index=False)


df1[df1.columns[0:]].corr()['Return_Cluster'][:]
'''
df.to_csv(r'/Users/johnnywang/Documents/Featured Engineering/data.csv',index=False)
df[df.columns[1:]].corr()['EURUSD_Cluster'][:]


import h2o
h2o.init()

filepath = r'/Users/johnnywang/Documents/Featured Engineering/data.csv'
data = h2o.import_file(filepath)
print('dimension:', data.shape)
data.head(6)

# Since the task we're dealing at hand is a binary classification problem, 
# we must ensure that our response variable is encoded as a factor type. 
# If the response is represented as numerical values of 0/1, 
# H2O will assume we want to train a regression model.
# Encode the binary reponse as a factor
label_col = 'EURUSD_Cluster'
data[label_col] = data[label_col].asfactor()

# This is an optional step that checks the factor level
data[label_col].levels()

# If we check types of each column, we can see which columns
# are treated as categorical type (listed as 'enum')
data.types

# Next, we perform a three-way split: 70% for training; 15% for validation; and 15% for final testing.
# We will train a data set on one set and use the others to test the validity of the model 
# by ensuring that it can predict accurately on data the model has not been shown, 
# i.e. to ensure our model is generalizable.
# 1. for the splitting percentage, we can leave off
# the last proportion, and h2o will generate the
# number for the last subset for us
# 2. setting a seed will guarantee reproducibility
random_split_seed = 1234
train, valid, test = data.split_frame([0.7, 0.15], seed = random_split_seed)
print(train.nrow)
print(valid.nrow)
print(test.nrow)

# Here, we extract the column name that will serve as our response and predictors. 
# These informations will be used during the model training phase.
# .names, .col_names, .columns are all equivalent way of accessing the list 
# of column names for the H2O dataframe.
input_cols = data.columns
# Remove the response and the interest rate column since it's correlated with our response.
input_cols.remove(label_col)
input_cols.remove('Date')
input_cols.remove('S&P')
input_cols.remove('S&P_Cluster')
input_cols.remove('STD')
input_cols.remove('EURUSD=X')
input_cols.remove('Vol_global')
input_cols.remove('Primary risk')
input_cols.remove('FD')
input_cols.remove('Hurst')
input_cols.remove('Correlation')
input_cols.remove('Vol_local')
input_cols.remove('Energy')
input_cols.remove('Return')




from h2o.estimators.gbm import H2OGradientBoostingEstimator

# We specify an id for the model so we can refer to it more easily later
gbm = H2OGradientBoostingEstimator(
    seed = 1,
    ntrees = 500,
    model_id = 'gbm1',
    stopping_rounds = 3,
    stopping_metric = 'auto',  #AUTO,MSE,
    score_tree_interval = 5,
    stopping_tolerance = 0.0005)

# Note that it is .train (not .fit) to train the model. Just in case you're coming from scikit-learn
gbm.train(
    y = label_col,
    x = input_cols,
    training_frame = train,
    validation_frame = valid)

# Evaluating the performance, printing the whole model performance object will 
# give us a whole bunch of information, we'll only be accessing the auc metric here.
gbm_test_performance = gbm.model_performance(test)
gbm_test_performance

gbm_history = gbm.scoring_history()
gbm_history


#mse 0.36482161681781133

#rmse  0.6040046496657218
plt.rcParams['figure.figsize'] = 8, 6
plt.rcParams['font.size'] = 12

plt.plot(gbm_history['training_rmse'], label = 'training_rmse')
plt.plot(gbm_history['validation_rmse'], label = 'validation_rmse')
plt.xticks(range(gbm_history.shape[0]), gbm_history['number_of_trees'].apply(int))
plt.title('GBM training history')
plt.legend()
plt.show()

#logloss  1.0170423053082867
plt.rcParams['figure.figsize'] = 8, 6
plt.rcParams['font.size'] = 12

plt.plot(gbm_history['training_logloss'], label = 'training_logloss')
plt.plot(gbm_history['validation_logloss'], label = 'validation_logloss')
plt.xticks(range(gbm_history.shape[0]), gbm_history['number_of_trees'].apply(int))
plt.title('GBM training history')
plt.legend()
plt.show()

#classification error 0.742334640812324
plt.rcParams['figure.figsize'] = 8, 6
plt.rcParams['font.size'] = 12

plt.plot(gbm_history['training_classification_error'], label = 'training_classification_error')
plt.plot(gbm_history['validation_classification_error'], label = 'validation_classification_error')
plt.xticks(range(gbm_history.shape[0]), gbm_history['number_of_trees'].apply(int))
plt.title('GBM training history')
plt.legend()
plt.show()


# Hyperparameter Tuning
# When training machine learning algorithm, often times we wish to perform hyperparameter search. 
# Thus rather than training our model with different parameters manually one-by-one, we will make use of 
# the H2O's Grid Search functionality. H2O offers two types of grid search: Cartesian and RandomDiscrete. 
# Cartesian is the traditional, exhaustive, grid search over all the combinations of model parameters in the grid, 
# whereas Random Grid Search will sample sets of model parameters randomly for some specified period of time (or maximum number of models).
# Specify the grid

gbm_params = {
    'max_depth': [3, 5, 9],
    'sample_rate': [0.8, 1.0],
    'col_sample_rate': [0.2, 0.5, 1.0]}

# If we wish to specify model parameters that are not part of our grid, 
# we pass them along to the grid via the H2OGridSearch.train() method. 
from h2o.grid.grid_search import H2OGridSearch
gbm_tuned = H2OGridSearch(
    grid_id = 'gbm_tuned1',
    hyper_params = gbm_params,
    model = H2OGradientBoostingEstimator)
gbm_tuned.train(
    y = label_col,
    x = input_cols,
    training_frame = train,
    validation_frame = valid,
    # nfolds = 5,  # alternatively, we can use N-fold cross-validation
    ntrees = 100,
    stopping_rounds = 3,
    stopping_metric = 'mse',
    score_tree_interval = 5,
    stopping_tolerance = 0.0005)  # we can specify other parameters like early stopping here

# To compare the model performance among all the models in a grid, 
# sorted by a particular metric (e.g. AUC), we can use the get_grid method.
gbm_tuned = gbm_tuned.get_grid(
    sort_by = 'mse', decreasing = True)
gbm_tuned

# Instead of running a grid search, the example below shows the code modification needed to run a random search.
# In addition to the hyperparameter dictionary, we will need specify the search_criteria as 
# RandomDiscrete with a number for max_models, which is equivalent to the number of 
# iterations to run for the random search. This example is set to run fairly quickly, 
# we can increase max_models to cover more of the hyperparameter space. Also, we can expand 
# the hyperparameter space of each of the algorithms by modifying the hyperparameter list below.
# Specify the grid and search criteria: 
gbm_params = {
    'max_depth': [3, 5, 9],
    'sample_rate': [0.8, 1.0],
    'col_sample_rate': [0.2, 0.5, 1.0]}

# Note that in addition to max_models, we can specify max_runtime_secs
# to run as many model as we can for X amount of seconds. 
search_criteria = {
    'max_models': 5,
    'strategy': 'RandomDiscrete'}

# Train the hyperparameter searched model:
gbm_tuned = H2OGridSearch(
    grid_id = 'gbm_tuned2',
    hyper_params = gbm_params,
    search_criteria = search_criteria,
    model = H2OGradientBoostingEstimator)
gbm_tuned.train(
    y = label_col,
    x = input_cols,
    training_frame = train,
    validation_frame = valid,
    ntrees = 100)

# Evaluate the model performance: 
gbm_tuned = gbm_tuned.get_grid(
    sort_by = 'mse', decreasing = True)
gbm_tuned

# Lastly, let's extract the top model, as determined by the validation data's AUC score 
# from the grid and use it to evaluate the model performance on a test set, 
# so we get an honest estimate of the top model performance.

# Our model is reordered based on the sorting done above; hence we can retrieve 
# the first model id to retrieve the best performing model we currently have.
gbm_best = gbm_tuned.models[0]
gbm_best_performance = gbm_best.model_performance(test)
gbm_best_performance.mse()

# Saving and loading the model
model_path = h2o.save_model(
    model = gbm_best, path = 'h2o_gbm', force = True)
saved_model = h2o.load_model(model_path)
gbm_best_performance = saved_model.model_performance(test)
gbm_best_performance.mse()

# Model Interpretation
# After building our predictive model, often times we would like to inspect which variables/features 
# were contributing the most. This interpretation process allows us the double-check to ensure 
# what the model is learning makes intuitive sense and enable us to explain the results 
# to non-technical audiences. With h2o's tree-base models, we can access the 
# varimp attribute to get the top important features.
gbm_best.varimp(use_pandas = True)

# We can return the variable importance as a pandas dataframe, hopefully the table 
# should make intuitive sense, where the first column is the feature/variable and 
# the rest of the columns are the feature's importance represented under different scale. 
# We'll be working with the last column, where the feature importance has been normalized 
# to sum up to 1. And note the results are already sorting in decreasing order, 
# where the more important the feature the earlier the feature will appear in the table.




###################################### Vanilla LSTM ########################################
import random
from sklearn.metrics import mean_squared_error

#After we select the feature, we remove the features which are not so important. 
input_cols.remove('Vol_local_Cluster')
input_cols.remove('Energy_Cluster')
input_cols.remove('Return_Cluster')

#we still have 6 features remaining
n_steps = 6
# reshape from [samples, timesteps] into [samples, timesteps, features]
X, y = data[input_cols].as_data_frame().values, data[label_col].as_data_frame().values

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = X
x_input = x_input.reshape((X.shape[0], X.shape[1], n_features))
yhat = model.predict(x_input, verbose=0)

#error is the difference between the true cluster and the prediction.
error1=abs(yhat-y)
average_error1 = sum(error1)/len(error1) # the same of the errors divide the number of days

# we plot the scatter graph to see the accuracy of the prediction, the more they overlap, the more accurate it will be. 
plt.scatter(range(len(y)), y, s=15)
plt.scatter(range(len(yhat)), yhat, s=15)
plt.title('Vanilla LSTM')
plt.xlabel('Day')
plt.ylabel('Cluster')
plt.show()




###################################### Stacked LSTM ########################################

# choose a number of time steps
n_steps = 6
# split into samples
X, y = data[input_cols].as_data_frame().values, data[label_col].as_data_frame().values
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = X
x_input = x_input.reshape((X.shape[0], X.shape[1], n_features))
yhat = model.predict(x_input, verbose=0)

error2=abs(yhat-y)
average_error2 = sum(error2)/len(error2)

plt.scatter(range(len(y)), y, s=15)
plt.scatter(range(len(yhat)), yhat, s=15)
plt.title('Stacked LSTM')
plt.xlabel('Day')
plt.ylabel('Cluster')
plt.show()


###################################### Bidirectional LSTM ########################################


# choose a number of time steps
n_steps = 6
# split into samples
X, y = data[input_cols].as_data_frame().values, data[label_col].as_data_frame().values
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = X
x_input = x_input.reshape((X.shape[0], X.shape[1], n_features))
yhat = model.predict(x_input, verbose=0)

error3=abs(yhat-y)
average_error3 = sum(error3)/len(error3)

plt.scatter(range(len(y)), y, s=15)
plt.scatter(range(len(yhat)), yhat, s=15)
plt.title('Bidirectional LSTM')
plt.xlabel('Day')
plt.ylabel('Cluster')
plt.show()

###################################### CNN LSTM ########################################

# choose a number of time steps
n_steps = 6
# split into samples
X, y = data[input_cols].as_data_frame().values, data[label_col].as_data_frame().values
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 3
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input = X
x_input = x_input.reshape((X.shape[0], n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)

error4=abs(yhat-y)
average_error4 = sum(error4)/len(error4)

plt.scatter(range(len(y)), y, s=15)
plt.scatter(range(len(yhat)), yhat, s=15)
plt.title('CNN LSTM')
plt.xlabel('Day')
plt.ylabel('Cluster')
plt.show()

###################################### ConvLSTM ########################################

# choose a number of time steps
n_steps = 6
# split into samples
X, y = data[input_cols].as_data_frame().values, data[label_col].as_data_frame().values
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 2
n_steps = 3
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
# define model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input = X
x_input = x_input.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)

error5=abs(yhat-y)
average_error5 = sum(error5)/len(error5)

plt.scatter(range(len(y)), y, s=15)
plt.scatter(range(len(yhat)), yhat, s=15)
plt.title('ConvLSTM')
plt.xlabel('Day')
plt.ylabel('Cluster')
plt.show()

###################################### Multiple Input Series ########################################
    
# To add a parallel feature, we will caluculate the data of the other currency to clusters. 
file = '/Users/johnnywang/Documents/Featured Engineering/Currencies_and_SP500(3).csv'
# Set rolling window range
range_period_list = list(range(150,160,10))
# Read file
df = pd.read_csv(file,
                 names=['Date', 'EURMXN=X','S&P','EURMXN_Cluster','S&P_Cluster','STD', 'STD_Cluster','Vol_global', 
                        'Vol_global_Cluster','Vol_local', 'Vol_local_Cluster','Primary risk', 'Primary risk_Cluster',
                        'FD','FD_Cluster','Hurst', 'Hurst_Cluster','Correlation', 'Correlation_Cluster','Energy',
                        'Energy_Cluster','Return', 'Return_Cluster'])
regressor_list = []    

def Clusters (Data):
    
        Data = Data.fillna(0)

        X = Data.values.reshape(-1,1)

# initialize k-means algo object with 4 clusters

        kmeans = KMeans(n_clusters=4, init='k-means++', tol=0.004, max_iter=300, random_state=42).fit(X) #get kmeans

        y_kmeans = kmeans.predict(X)


        return y_kmeans
    
for range_period in range_period_list:
    """
    Calculation
    """
    # range_period = 150 #//the best window in this condition - see lines 115-130: time window optimization
    # calculate the std within range 
    df['STD'] = df['EURMXN=X'].rolling(range_period).std(ddof=0)

    # get the std_max to calculate the vol_global
    df_std_max = df['STD'].max()
    # calculate the vol_global
    df['Vol_global'] = df['STD'] / df_std_max

    # calculate the vol_local
    df['Vol_local'] = df['STD'] / df['STD'].rolling(10).mean() / 3

    # calculate the primary risk using np.ceil (round up)
    df['Primary risk'] = (1 / df['Vol_global']).apply(np.ceil) / 10

    # calculate the correlation
    df['Correlation'] = 0.5 * (df['EURMXN=X'].rolling(range_period).corr(df['S&P']) + 1)

    # Calculate FD and Hurst exponent
    FDs = list()
    Hurst = list()
    for i, data_row in df.iterrows():
        if i < len(df) + 1 - range_period:
            spy_close = df[i:i + range_period - 1]
            spy_close = spy_close[['EURMXN=X']].copy()
            lag1, lag2 = 2, 20  # lag chosen 2 , 20
            lags = range(lag1, lag2)

            # Fractal Dimension calculation
            tau = [sqrt(std(subtract(spy_close[lag:], spy_close[:-lag]))) for lag in lags]
            m = polyfit(log(lags), log(tau), 1)
            hurst = m[0] * 2
            fractal = 1 - hurst[0]
            FDs.append(fractal)
            Hurst.append(hurst)
        else:
            break
    df['FD'][range_period - 1:len(df) + 1] = FDs
    df['Hurst'][range_period - 1:len(df) + 1] = Hurst

    # Linear regression to find coeff for calculating the energy
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    Vols = df['Vol_global'][range_period - 1:len(df) + 1]
    FD = df['FD'][range_period - 1:len(df) + 1]
    X = Vols.values.reshape(-1, 1)
    y = FD.values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Linear Regression model built on train data
    regressor = LinearRegression()

    # Training the algorithm
    regressor.fit(X_train, y_train)

    # Calculating the coeff B of Fractal Dim
    c = regressor.intercept_
    B = 1 / c

    # Calculating the coeff A of Vol
    m = regressor.coef_
    A = m.item() * B
   

    # Calculate the Energy
    df['Energy'] = A * df['Vol_global'] + B * df['FD']
    energy_list = list(df['Energy'].values)
    energy_list = energy_list[range_period-1:]
    response_list = np.array(range(len(energy_list)))
    #fit the linear regression between energy and time
    regressor.fit(np.array(energy_list).reshape(-1,1),response_list.reshape(-1,1))
    regressor_list.append(regressor)
    
    df['Return'] = (df['EURMXN=X'] - df['EURMXN=X'].shift(1)) / df['EURMXN=X'].shift(1)

    df['Return'] = df['Return'].fillna(0)
    
    #we use the cluster function to make all features into clusters
    df['EURMXN_Cluster']=Clusters(df['EURMXN=X'])
    df['S&P_Cluster']=Clusters(df['S&P'])
    df['STD_Cluster']=Clusters(df['STD'])
    df['Vol_global_Cluster']=Clusters(df['Vol_global'])
    df['Vol_local_Cluster']=Clusters(df['Vol_local'])
    df['Primary risk_Cluster']=Clusters(df['Primary risk'])
    df['FD_Cluster']=Clusters(df['FD'])
    df['Hurst_Cluster']=Clusters(df['Hurst'])
    df['Correlation_Cluster']=Clusters(df['Correlation'])
    df['Energy_Cluster']=Clusters(df['Energy'])
    df['Return_Cluster']=Clusters(df['Return'])


df.to_csv(r'/Users/johnnywang/Documents/Featured Engineering/data2.csv',index=False)
df[df.columns[1:]].corr()['EURMXN_Cluster'][:]


filepath = r'/Users/johnnywang/Documents/Featured Engineering/data2.csv'
data1 = h2o.import_file(filepath)
print('dimension:', data1.shape)
data1.head(6)

input_cols1 = data1.columns
# Remove the response and the interest rate column since it's correlated with our response.
input_cols1.remove('EURMXN_Cluster')
input_cols1.remove('Date')
input_cols1.remove('S&P')
input_cols1.remove('S&P_Cluster')
input_cols1.remove('STD')
input_cols1.remove('EURMXN=X')
input_cols1.remove('Vol_global')
input_cols1.remove('Primary risk')
input_cols1.remove('FD')
input_cols1.remove('Hurst')
input_cols1.remove('Correlation')
input_cols1.remove('Vol_local')
input_cols1.remove('Energy')
input_cols1.remove('Return')
input_cols1.remove('Vol_local_Cluster')
input_cols1.remove('Energy_Cluster')
input_cols1.remove('Return_Cluster')



def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = data[input_cols].as_data_frame().values
in_seq2 = data1[input_cols1].as_data_frame().values
out_seq = data[label_col].as_data_frame().values
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 6))
in_seq2 = in_seq2.reshape((len(in_seq2), 6))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 6
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = X
x_input = x_input.reshape((X.shape[0], n_steps, n_features))
yhat = model.predict(x_input, verbose=0)


y=y.reshape(3769,1)
# calculate the difference and average difference
error6=abs(yhat-y)
average_error6 = sum(error6)/len(error6)

# plot the scatter plot
plt.scatter(range(len(y)), y, s=15)
plt.scatter(range(len(yhat)), yhat, s=15)
plt.title('Multiple Input Series')
plt.xlabel('Day')
plt.ylabel('Cluster')

plt.show()


###################################### Multiple Input Multi-Step Output ########################################

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
in_seq1 = data[input_cols].as_data_frame().values
in_seq2 = data1[input_cols1].as_data_frame().values
out_seq = data[label_col].as_data_frame().values
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 6))
in_seq2 = in_seq2.reshape((len(in_seq2), 6))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 6, 4
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = X
x_input = x_input.reshape((X.shape[0], n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)

#trends is to see if the next 3 days are consecutively increasing or decreasing. 
#1 means upwards, -1 means downwards, 0 means no specific up or down
trend=[]
for i in range(0,len(yhat)):
    if yhat[i][0]<yhat[i][1]<yhat[i][2]<yhat[i][3]:
        trend.append(1)
    elif yhat[i][0]>yhat[i][1]>yhat[i][2]>yhat[i][3]:
        trend.append(-1)
    else:
        trend.append(0)



