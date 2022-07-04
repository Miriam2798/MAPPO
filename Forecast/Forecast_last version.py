import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
import math
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import date
from sklearn.preprocessing import LabelEncoder
from itertools import islice
import os
import requests
#Nota: Tengo que hacer limpieza de imports 

#ACCES A SERVIDOR: 
    #avg_no2 --> avg
    #detailed_no2 --> hores
option_db = 'avg_no2'
pollutant = requests.get('http://mappo-server.herokuapp.com/database?option='+option_db)


pollutant = str(pollutant.content)
primer_caracter = False
last_char = ''
#print(pollutant)
with open(option_db+'.csv', 'w') as f:

    if "detailed" not in option_db:
        f.write("estacio,data,contaminant,unitats,latitud,longitud,valor,\n")
    else:
        f.write("estacio,data,contaminant,unitats,latitud,longitud,quantity,\n")

    for c in pollutant:
        if c == '(':
            primer_caracter = True
            
        if (c >= '0' and c <= '9' or c >= 'A' and c <= 'Z' or c >= 'a' and c <= 'z' or c == '.' or c == '-' or c == ',' or c == ' ' or c == ':') and primer_caracter==True:
            last_char = c
            f.write(c)

        elif c == ')' and primer_caracter == True:
            f.write(',')
            last_char = c
            f.write('\n')
            primer_caracter = False

#DATASET
dataset = pd.read_csv('/content/avg_no2.csv', sep = ',')#index_col=1
df = pd.DataFrame(dataset, columns=['estacio','data', 'contaminant', 'unitats', 'latitud', 'longitud', 'valor'])
df.fillna(method="ffill", inplace=True)
df = df.sort_values(by='data')

#PREPROCESADO
valores = df['valor']
dates = df['data']
suma = np.zeros(len(valores), float)
dates.reset_index(drop=True, inplace=True)
valores.reset_index(drop=True, inplace=True)
fecha = []
#print(dates)

# Computing daily avg 
multiple = 8
ultim = 943
for i in range(len(valores)):
  if i%8 == 0 and i!=0 or i==ultim:    
    if i==ultim:
      fecha.append(dates[i-multiple+1]) 
      suma[i-multiple] = (sum(valores[(i-multiple+1):(i+1)])/multiple)
    else:
      fecha.append(dates[i-multiple])
      suma[i-multiple] = (sum(valores[(i-multiple):i])/multiple)

avg = suma[suma!=0]
#print((avg))
#print(fecha)

dfd = pd.DataFrame({'fecha': fecha, 'avg': list(avg)}, index=fecha, columns=['avg'])
print(dfd.size)

#PERSISTENCE MODEL 
#Def
# convert an array of values into a dataset matrix
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# transform series into train and test sets for supervised learning
def prepare_data(raw_values, n_test, n_lag, n_seq):
	# extract raw values
	#raw_values = series.values
	raw_values = raw_values.reshape(len(raw_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(raw_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return train, test

  # make a persistence forecast
def persistence(last_ob, n_seq):
	return [last_ob for i in range(n_seq)]

# evaluate the persistence model
def make_forecasts(train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = persistence(X[-1], n_seq)
		# store the forecast
		forecasts.append(forecast)
	return forecasts

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = test[:,(n_lag+i)]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = math.sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))
 
# plot the forecasts in the context of the original dataset
def plot_forecasts(values, forecasts, n_test):
	# plot the entire dataset in blue
	plt.plot(values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(values) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)] 
		yaxis = [values[off_s]] + forecasts[i]
		plt.plot(xaxis, yaxis, color='red')
	# show the plot
	plt.show()

#Calcul 
values = dfd.values

# integer encode direction
encoder = LabelEncoder()
values[:,0] = encoder.fit_transform(values[:,0])

# ensure all data is float
values = values.astype('float32')

# normalize dataset
scaler = MinMaxScaler(feature_range=(0,1))
#scaled = scaler.fit_transform(df['quantity'].array.reshape(-1,1)) #reshape(-1,1) porque sólo es una feature
scaled = scaler.fit_transform(values)

# configure
n_lag = 8
n_seq = 1
n_test = 18
# prepare data
train, test = prepare_data(scaled, n_test, n_lag, n_seq)
print('Train: %s, Test: %s' % (train.shape, test.shape))
 
# make forecasts
forecasts = make_forecasts(train, test, n_lag, n_seq)
# evaluate forecasts
evaluate_forecasts(test, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(scaled, forecasts, n_test+2)
