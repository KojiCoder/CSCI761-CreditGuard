import pandas as pd
import numpy as np
from math import radians
from sklearn.preprocessing import StandardScaler

important_cols = ['cc_num', 'category', 'amt', 'lat', 'long', 'city_pop', 'dob', 'unix_time', 'merch_lat', 'merch_long', 'trans_date_trans_time', 'is_fraud']
# store_fraud_status = ['is_fraud']

# #generate a separate csv that stores whether or not a transaction is frauduent
# fraud_data = pd.read_csv('dataset/fraudTrainOriginal.csv', usecols=store_fraud_status)
# fraud_data.to_csv('dataset/fraudTrainFraudRecord.csv', index=False)

#preprocessing the data
pp_data = pd.read_csv('dataset/fraudTrainOriginal.csv', usecols=important_cols)
pp_data['trans_datetime']   = pd.to_datetime(pp_data['trans_date_trans_time'])
pp_data['hour']             = pp_data['trans_datetime'].dt.hour
pp_data['day_of_week']      = pp_data['trans_datetime'].dt.dayofweek
pp_data['month']            = pp_data['trans_datetime'].dt.month

#remove the date data
pp_data.drop(columns=['trans_date_trans_time', 'trans_datetime'], inplace=True)

#get age from data
pp_data['dob'] = pd.to_datetime(pp_data['dob'])
ref_date = pd.to_datetime('2019-01-01')
pp_data['age'] = (ref_date - pp_data['dob']).dt.days // 365
pp_data.drop(columns=['dob'], inplace=True)

#get distance between cardholder and merchant
def haversine_vec(lat1, long1, lat2, long2):
    #Earth's radius in km
    R = 6371
    lat1, long1, lat2, long2 = map(np.radians, [lat1, long1, lat2, long2])
    diff_lat = lat2 - lat1
    diff_long = long2 - long1
    a = np.sin(diff_lat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(diff_long/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

pp_data['dist_to_merch'] = haversine_vec(pp_data['lat'], pp_data['long'], pp_data['merch_lat'], pp_data['merch_long'])
pp_data.drop(columns = ['lat', 'long', 'merch_lat', 'merch_long'], inplace=True)

#scale numerical features
scale_cols = ['amt', 'city_pop', 'unix_time', 'age', 'dist_to_merch', 'hour', 'day_of_week', 'month']
scaler = StandardScaler()
pp_data[scale_cols] = scaler.fit_transform(pp_data[scale_cols])

#data that only has normal transactions used to train the autoencoder
pp_normal_trans = pp_data[pp_data['is_fraud'] == 0].drop(columns=['is_fraud'])

pp_data.to_csv('dataset/fraudTrain.csv', index=False)
pp_normal_trans.to_csv('dataset/normalTrain.csv', index=False)

#build sequences for LSTM
seq_len = 10
data_sorted_unixtime = pp_data.sort_values('unix_time')
sequences = []
labels = []

for card_id, group, in data_sorted_unixtime.groupby('cc_num'):
    #drops cc_num and is_fraud columns before feeding to the LSTM
    features = group.drop(columns=['cc_num', 'is_fraud']).values
    target = group['is_fraud'].values
    for i in range(len(features) - seq_len):
        sequences.append(features[i:i+seq_len])
        labels.append(target[i+seq_len])

x_seq = np.array(sequences)
y_seq = np.array(labels)

