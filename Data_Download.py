# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 08:50:39 2020

@author: Zachary
"""

# Load required libraries
from climata.snotel import StationDailyDataIO
from climata.snotel import StationIO
import pandas as pd
import numpy as np
from functools import reduce
import time
import datetime
import pickle

# Define function for accessing NRCS API and downloading individual time series data files
def getNRCS(station_id, param_id, nyears, frequency):
  ndays, yesterday = 365*nyears, datetime.date.today()
  datelist = pd.date_range(end=yesterday, periods=ndays).tolist()
  
  data = StationDailyDataIO(
      start_date=datelist[0],
      end_date=datelist[-1],
      station=station_id,
      parameter=param_id,
  )
  if len(data.data) == 0:
    return print('The data source is empty for this location')
  temp = pd.DataFrame(data.data[0]['data'].data)
  df = pd.DataFrame(temp['value'], columns = ['value'])
  df.index = pd.to_datetime(temp['date'])
  df.index.name = 'Date'
  if df.index[-1].year != datetime.date.today().year:
    return print('Either today is new years, or the gap in data is too large')
  if df.index[0].year > 1990:
    return print('The starting year in the time series is too recent')
  df.columns = [station_id]
  missing_data = len(df)-len(df.dropna())
  if missing_data > 100:
    return print('Today is definietly not new years, but the gap in the data is still too large')
  print('Missing data points:', missing_data)
  if missing_data > 0:
    new = pd.DataFrame()
    cols = df.columns
    interp_df = [new.assign(Interpolant = df[i].interpolate(method='time')) for i in cols][0]
    interp_df.columns = [station_id]
    resampled_df = interp_df.resample(frequency).mean()
    return resampled_df
  else:
    resampled_df = df.resample(frequency).mean()
    return resampled_df

# Define function for performing a bulk download of all time series files from NRCS API based on a datatype (e.g. snow water equivalent (SWE))
def bulk_download(parameter, years, frequency):
	# Download snow water equivalent data across central and northern Utah
	stations = StationIO(state = 'UT', parameter = parameter, min_latitude = 40.3, max_latitude = 40.8, min_longitude = -111.90, max_longitude = -110.45)
	station_ids = [stations.data[i]['stationTriplet'] for i in range(len(stations))]
	station_names = [stations.data[i]['name'] for i in range(len(stations))]
	dflist = [getNRCS(station_ids[i], parameter, years, frequency) for i in range(len(stations)) if type(getNRCS(station_ids[i], parameter, years, frequency)) != type(None)]
	df = reduce(lambda x, y: pd.merge(x, y, on = 'Date'), dflist)
	# Populate metadata
	metadata = pd.DataFrame(station_ids)
	metadata.columns = ['id']
	metadata['name'] = station_names
	metadata['lat'] = [stations.data[i]['latitude'] for i in range(len(stations))]
	metadata['lng'] = [stations.data[i]['longitude'] for i in range(len(stations))]
	metadata = metadata[metadata['id'].isin(df.columns)]
	df.columns = metadata['name']
	return df, metadata
# %%
# Define object for Upper Stillwater Reservoir storage volume (ac-ft) time series data 
sv = getNRCS('09278000:UT:BOR', 'RESC', 31, 'W')
sv.columns = ['Upper Stillwater']
# Keep only values from 1990 to the present date
sv = sv[sv.index.year >= 1990]
# Define object for all SWE monitoring stations surrounding Upper Stillwater Reservoir
swe, swe_metadata = bulk_download('WTEQ', 31, 'W')
# %%
# Combine SV and SWE together into single dataframe
data = reduce(lambda x, y: pd.merge(x, y, left_index = True, right_index = True), [sv, swe])
# Save dataframe object as pkl file to be quickly loaded in future training sessions
pickle_out = open("project_data.pkl","wb")
pickle.dump(data, pickle_out)
# %%
# Load pkl file back into work session
# Note: check to make sure the file path location is correct
loaded_data = pickle.load(open(r'C:\Users\Zachary\Documents\Project Data\project_data.pkl', 'rb'))