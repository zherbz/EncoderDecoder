# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:31:38 2019
@author: Zachary
"""
import pickle
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Add
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
sns.set_style("darkgrid")

# calculate the absolute percent error for each forecast step
def mape(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test))*100, np.abs((y_test - y_pred) / y_test)*100

def sliding_window_input(df, input_length, idx):
	window = df.values[idx:input_length+idx,:]
	return window

def sliding_window_output(df, input_length, output_length, idx):
	window = df.values[input_length+idx:input_length+output_length+idx]
	return window

def model_configs():
	epochs = [100]
	batch_size = [32]
	n_nodes1 = [64]
	n_nodes2 = [32]
	filter1 = [16]
	filter2 = [32]
	kernel_size = [6]
	configs = []
	for a in epochs:
		for b in batch_size:
			for c in n_nodes1:
				for d in n_nodes2:
					for e in filter1:
						for f in filter2:
							for g in kernel_size:
								cfg = [a,b,c,d,e,f,g]
								configs.append(cfg)
	print('Total configs: %d' % len(configs))
	return configs

def build_model(data, config, output_length, X_test, input_length, index, repeats):
	print('Run:', str(index), '/', str(repeats))
	# start time
	start = time.time()
	print('Configuration:', config)
	# define parameters
	epochs, batch_size, n_nodes1, n_nodes2, filter1, filter2, kernel_size = config
	# convert data to supervised learning environment
	X_train = np.stack([sliding_window_input(data, input_length, i) for i in range(len(data)-input_length)])[:-output_length]
	y_train = np.stack([sliding_window_output(data['Upper Stillwater'], input_length, output_length, i) for i in range(len(data)-(input_length+output_length))])
	y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
	print('Feature Training Tensor (samples, timesteps, features):', X_train.shape)
	print('Target Training Tensor (samples, timesteps, features):', y_train.shape)
	# define parameters
	n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
	# define residual CNN-LSTM
	visible1 = Input(shape = (n_timesteps,n_features))
	
	model = Conv1D(filter1, kernel_size, padding = 'causal')(visible1)
	residual1 = ReLU()(model)
	model = Conv1D(filter1, kernel_size, padding = 'causal')(residual1)
	model = Add()([residual1, model])
	model = ReLU()(model)
	
	model = Conv1D(filter1, kernel_size, padding = 'causal')(model)
	residual2 = ReLU()(model)
	model = Conv1D(filter1, kernel_size, padding = 'causal')(residual2)
	model = Add()([residual2, model])
	model = ReLU()(model)
	model = MaxPooling1D()(model)
	
	model = Conv1D(filter2, kernel_size, padding = 'causal')(model)
	residual3 = ReLU()(model)
	model = Conv1D(filter2, kernel_size, padding = 'causal')(residual3)
	model = Add()([residual3, model])
	model = ReLU()(model)
	
	model = Conv1D(filter2, kernel_size, padding = 'causal')(model)
	residual4 = ReLU()(model)
	model = Conv1D(filter2, kernel_size, padding = 'causal')(residual4)
	model = Add()([residual4, model])
	model = ReLU()(model)
	model = MaxPooling1D()(model)
	
	model = Conv1D(n_nodes1, kernel_size, padding = 'causal')(model)
	residual5 = ReLU()(model)
	model = Conv1D(n_nodes1, kernel_size, padding = 'causal')(residual5)
	model = Add()([residual5, model])
	model = ReLU()(model)
	
	model = Conv1D(n_nodes1, kernel_size, padding = 'causal')(model)
	residual6 = ReLU()(model)
	model = Conv1D(n_nodes1, kernel_size, padding = 'causal')(residual6)
	model = Add()([residual6, model])
	model = ReLU()(model)
	model = MaxPooling1D()(model)
	
	model = Flatten()(model)
	model = RepeatVector(n_outputs)(model)
	
	model = LSTM(n_nodes1, activation='relu', return_sequences=True)(model)
	model = LSTM(n_nodes1, activation='relu', return_sequences=True)(model)
	model = LSTM(n_nodes1, activation='relu', return_sequences=True)(model)
	model = LSTM(n_nodes1, activation='relu', return_sequences=True)(model)
	
	dense = TimeDistributed(Dense(n_nodes2, activation='relu'))(model)
	output = TimeDistributed(Dense(1))(dense)
	model = Model(inputs = visible1, outputs = output)
	model.compile(loss='mse', optimizer='adam')
	#model.summary()
	# fit network
	es = EarlyStopping(monitor='loss', mode='min', patience = 10)
	history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2, callbacks = [es])
	# test model against hold-out set
	y_pred = model.predict(X_test)
	print('\n Elapsed time:', round((time.time() - start)/60,3), 'minutes')
	return y_pred, history

def get_stats(y_pred, y_test):
	print('Mean Absolute Error:', round((mean_absolute_error(y_test, y_pred)/np.mean(y_test))*100,3), '%')
	print('Root Mean Squared Error:', round(((mean_squared_error(y_test, y_pred)**0.5)/np.mean(y_test))*100,3), '%')
	print('Median Absolute Error:', round((median_absolute_error(y_test, y_pred)/np.mean(y_test))*100,3), '%')
	print('NSE:', round(r2_score(y_test, y_pred), 3))
	print('Explained Variance:', round(explained_variance_score(y_test, y_pred),3), '\n')
	
def get_stats_df(y_pred, y_test):
	Meanae = str(round((mean_absolute_error(y_test, y_pred)/np.mean(y_test))*100,3))
	RMSE = str(round(((mean_squared_error(y_test, y_pred)**0.5)/np.mean(y_test))*100,3))
	Medae = str(round((median_absolute_error(y_test, y_pred)/np.mean(y_test))*100,3))
	NSE = str(round(r2_score(y_test, y_pred), 3))
	Expvar = str(round(explained_variance_score(y_test, y_pred),3))
	stats = [[Meanae], [RMSE], [Medae], [NSE], [Expvar]]
	index = ['Mean Absolute Error (%)', 'Root Mean Squared Error (%)', 'Median Absolute Error (%)', 'NSE', 'Explained Variance']
	return stats, index

def ape(true, test):
	value = (abs(true - test)/true)*100
	return np.array(value)

def repeat_runs(repeats, data, cfg, output_length, X_test, input_length, y_test):
	print('Start of Training Period:', data.index[0])
	print('End of Training Period:', data.index[-1], '\n')
	output = [build_model(data, cfg, output_length, X_test, input_length, i, repeats) for i in range(1, repeats+1)]
	y_pred = [sv_scaler.inverse_transform(output[i][0][0].reshape(-1,1)) for i in range(len(output))]
	
	loss = [pd.DataFrame(output[i][1].history['loss']) for i in range(len(output))]
	val_loss = [pd.DataFrame(output[i][1].history['val_loss']) for i in range(len(output))]
	max_epochs = max([len(loss[i]) for i in range(len(loss))])
	loss = [loss[i].values for i in range(len(loss))]
	val_loss = [val_loss[i].values for i in range(len(val_loss))]
	for i in range(len(loss)):
		if len(loss[i]) == max_epochs:
			loss[i] = loss[i]
		else:
			while len(loss[i]) < max_epochs:
				loss[i] = np.append(loss[i], np.nan)
				val_loss[i] = np.append(val_loss[i], np.nan)
		loss[i] = pd.DataFrame(loss[i])
		val_loss[i] = pd.DataFrame(val_loss[i])
	loss_df = reduce(lambda x, y: pd.merge(x, y, left_index = True, right_index = True), loss)
	val_loss_df = reduce(lambda x, y: pd.merge(x, y, left_index = True, right_index = True), val_loss)
	
	loss_avg = loss_df.mean(axis = 1)
	loss_avg.name = 'loss_avg'
	val_loss_std = val_loss_df.std(axis = 1)
	val_loss_std.name = 'val_loss_std'
	val_loss_avg = val_loss_df.mean(axis = 1)
	val_loss_avg.name = 'val_loss_avg'
	upper_val = pd.Series(val_loss_avg.values + 1.96*val_loss_std.values/(repeats**(0.5)))
	upper_val.name = 'upper_val'
	lower_val = pd.Series(val_loss_avg.values - 1.96*val_loss_std.values/(repeats**(0.5)))
	lower_val.name = 'lower_val'
	history = reduce(lambda x, y: pd.merge(x, y, left_index = True, right_index = True), [loss_avg, val_loss_avg, upper_val, lower_val])
	history.columns = ['loss_avg', 'val_loss_avg', 'upper_val', 'lower_val']
	
	plt.figure(figsize=(20,10))
	plt.plot(history.index, history.loss_avg.values, color='Blue', linewidth=2, linestyle='solid', label="Training Loss (80%)")
	plt.plot(history.index, history.val_loss_avg.values, color='Orange', linewidth=2, linestyle='solid', label="Validation Loss (20%)")
	plt.fill_between(history.index, history.upper_val, history.lower_val, color='Orange', alpha=.15, label = '95% Confidence Interval')
	plt.title(str(y_test.index[0].year)+" Model Learning Curve", fontsize = 18)
	plt.ylabel("Loss: Mean Squared Error", fontsize=14)
	plt.xlabel("Experiences", fontsize=14)
	plt.legend(fontsize=14, loc = 'upper left');
	plt.savefig(fname = str(y_test.index[0].year)+'_lc.pdf',bbox_inches = 'tight');
	
	df = pd.DataFrame(np.concatenate(y_pred, axis = 1), index = y_test.index.strftime('%B %d'))
	spills = np.stack([sv.max() for i in range(len(y_test))])
	future_idx = range(1, len(y_test)+1)
	x_labels = y_test.index
	x_labels = x_labels.strftime('%B %d')
	[get_stats(y_test.values, y_pred[i]) for i in range(len(output))]
	
	q1, q3 = df.quantile(q = 0.25, axis = 1), df.quantile(q = 0.75, axis = 1)
	iqr = q3 - q1
	upper_bnd = q3 + 1.5*iqr
	lower_bnd = q1 - 1.5*iqr
	
	filtered_avg = np.stack([df.iloc[i][(df.iloc[i] < upper_bnd[i]) & (df.iloc[i] > lower_bnd[i])].mean() for i in range(len(df))])
	filtered_std = np.stack([df.iloc[i][(df.iloc[i] < upper_bnd[i]) & (df.iloc[i] > lower_bnd[i])].std() for i in range(len(df))])
	upper = filtered_avg + 1.96*filtered_std/(repeats**(0.5))
	lower = filtered_avg - 1.96*filtered_std/(repeats**(0.5))
	cellText, rows = get_stats_df(y_test.values, filtered_avg.reshape(-1,))
	
	plt.figure(figsize=(20,10))
	plt.boxplot(df)
	plt.xticks([i for i in future_idx],x_labels)
	plt.plot(future_idx, filtered_avg, color='Red', linewidth=2, linestyle='solid', label="Forecasted")
	plt.plot(future_idx, y_test.values, color='Green', linewidth=2, linestyle='solid', label="Actual")
	plt.plot(future_idx, spills, color = 'black', linewidth = 2, linestyle='dashed', label="Spill Limit")
	plt.fill_between(future_idx, upper, lower, color='k', alpha=.15, label = '95% Confidence Interval')
	plt.table(cellText = cellText, rowLabels = rows, colLabels = ['Stats'], bbox = [0.2,0.4,0.1,0.5])
	plt.title(str(y_test.index[0].year)+" Upper Stillwater Reservoir", fontsize = 18)
	plt.ylabel("Storage Volume (ac-ft)", fontsize=14)
	plt.xlabel("Future Time", fontsize=14)
	plt.legend(fontsize=14, loc = 'upper left');
	plt.savefig(fname = str(y_test.index[0].year)+'_fc.pdf',bbox_inches = 'tight');
	
	errors = np.stack(np.array([ape(y_test.values[i], df.mean(axis = 1).values[i]) for i in range(len(df))]))
	upper = np.stack(np.array([ape(y_test.values[i], df.mean(axis = 1).values[i] + (df.std(axis = 1).values[i]*1.96)/(repeats**0.5)) for i in range(len(df))]))
	lower = np.stack(np.array([ape(y_test.values[i], df.mean(axis = 1).values[i] - (df.std(axis = 1).values[i]*1.96)/(repeats**0.5)) for i in range(len(df))]))
	errors_df = pd.DataFrame(lower, columns = ['Lower Limit'])
	errors_df['Average'] = errors
	errors_df['Upper Limit'] = upper
	errors_df.index = df.index
	errors_df.plot(kind = 'bar', figsize = [20,10], title = 'Absolute Percent Error (%)');
	return df, history
# %%
data = pickle.load(open(r'C:\Users\Documents\Project Data\stillwater_data.pkl', 'rb'))
sv = data['Upper Stillwater']
swe = data[data.columns[1:]]
# Declare model input parameters
input_length, output_length, repeats, year = 20, 15, 30, 2019

# Assign range of values for model to be tested on
test_output = sv[(sv.index.year == year) & (sv.index.month >= 4)][:output_length]

# Assign range of values for model input during testing
test_input_start = test_output.index[0] - timedelta(weeks = input_length)
test_input = data[data.index >= test_input_start][:input_length]
test_input_end = test_input.index[-1]
training = data[data.index < test_input_start]

# Temporarily combine training and test_input back together for scaling purposes
data_for_scaling = pd.concat([training, test_input])

# Scale target vector seperately from additional features
sv_scaler = MinMaxScaler(feature_range=(0, 1))
sv_scaled = pd.DataFrame(sv_scaler.fit_transform(data_for_scaling['Upper Stillwater'].values.reshape(-1,1)), index = data_for_scaling.index)
variable_scaler = MinMaxScaler(feature_range=(0, 1))
variable_scaled = pd.DataFrame(variable_scaler.fit_transform(data_for_scaling[data_for_scaling.columns[1:]].values), index = data_for_scaling.index)
variable_scaled.columns = data.columns[1:]

# Combine scaled data back into single dataframe
scaled_data = pd.merge(sv_scaled, variable_scaled, left_index = True, right_index = True)
scaled_data.columns = data_for_scaling.columns

# Assign data to be used for model training
training = scaled_data[:-input_length]
training['Upper Stillwater'].plot(figsize = [20,10], title = 'Target Time Series: Upper Stillwater Reservoir Storage Volume (ac-ft)');
print('\n Selected Starting Point for Training:', training.index[0])
print('\n Selected Ending Point for Training:', training.index[-1])

# Assign data to be used for model test output (unbiased to model and scaler)
y_test = test_output
plt.figure(figsize=(20,10))
print('\n Selected Test Output: \n', y_test.plot(figsize = [20,10], title = 'Target Test Output'));

# Define snow water equivalent input for model
swe_input = variable_scaled[swe.columns][-input_length:]

fig, ax1 = plt.subplots()
# Plot the scaled swe input for the model
ax = swe_input.plot(kind='line', figsize = [20,15], linewidth = 3, ms = 12, fontsize = 16, ax = ax1)
markers = 2*[m for m, func in Line2D.markers.items() if func != 'nothing' and m not in Line2D.filled_markers]
for i, line in enumerate(ax.get_lines()):
	line.set_marker(markers[i])
ax.legend(ax.get_lines(), swe_input.columns, loc='best', fontsize = 16)
#plt.show()
swe_training = training[swe.columns]
swe_training = pd.merge(sv_scaled, swe_training, left_index = True, right_index = True)
# Plot storage volume input for model
sv_input = sv_scaled[-input_length:]
sv_input.plot(figsize = [20,15], legend = False, fontsize = 16)

# Structure the model test inputs as 3D tensors (unbiased to model)
scaled_test_input = pd.merge(sv_input, swe_input, left_index = True, right_index = True)
X_test = scaled_test_input.values.reshape(1,scaled_test_input.shape[0],scaled_test_input.shape[1])
print('Testing Input Tensor (samples, timesteps, features):', X_test.shape)
# %%
# Fit the model to the data repeatedly 
model_repeats = repeat_runs(repeats, training, model_configs()[0], output_length, X_test, input_length, y_test)
# %%
model_repeats[0].to_excel(str(year)+'_fc.xlsx')
model_repeats[1].to_excel(str(year)+'_lc.xlsx')