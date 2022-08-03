import requests
import json
import numpy as np
import pandas as pd
import boto3

# load spectrum data from s3
BUCKET_NAME = 'ramcell-test-spectra-source'
OBJECT_NAME = 'Glucose 500mW 10sec.csv'
s3 = boto3.client('s3')
target_data_path = 'inference_target_data.csv'
s3.download_file(BUCKET_NAME, OBJECT_NAME, target_data_path)
spectra_data = pd.read_csv(target_data_path, header=None)

# given parameter
minshift=100 #381.98
maxshift=1000 #1792.4
input_dim=1000

#unit_shift = (spectra_data[0].iloc[-1] - spectra_data[0].iloc[0])/(spectra_data[0].shape[0]-1)
unit_shift = (maxshift - minshift)/input_dim 
print("unit_shift = ",unit_shift)

# given spectrum data interpolation for NN input format

if minshift <= spectra_data[0].iloc[0]:
    left_pad_flag=1
    #interp_minshift=minshift
else:
    left_pad_flag=0
    #interp_minshift=spectra_data[0].iloc[0]
if spectra_data[0].iloc[-1] <= maxshift:  
    right_pad_flag=1
    #interp_maxshift=maxshift
else:
    right_pad_flag=0
    #interp_maxshift=spectra_data[0].iloc[-1]

#interp_shift = np.arange(interp_minshift,interp_maxshift,unit_shift)

interp_shift = np.arange(minshift,maxshift+0.0000001,unit_shift)

print(interp_shift.shape,interp_shift[0],interp_shift[-1])
if left_pad_flag :
    interp_min_ind = np.where((interp_shift>spectra_data[0].iloc[0] -unit_shift/2) 
                       & (interp_shift<spectra_data[0].iloc[0] +unit_shift/2))[0][0]
else :
    interp_min_ind = np.where((interp_shift>minshift -unit_shift/2) 
                   & (interp_shift<minshift +unit_shift/2))[0][0]
if right_pad_flag:  
    interp_max_ind = np.where((interp_shift>spectra_data[0].iloc[-1] -unit_shift/2) 
                   & (interp_shift<spectra_data[0].iloc[-1] +unit_shift/2))[0][0]
else :
    interp_max_ind = np.where((interp_shift>maxshift -unit_shift/2) 
               & (interp_shift<maxshift +unit_shift/2))[0][0]

print('given data - left : [{}]={}, given data - right : [{}]={}'.format(interp_min_ind,interp_shift[interp_min_ind],interp_max_ind,interp_shift[interp_max_ind]))

interp_data = np.interp(interp_shift[interp_min_ind:interp_max_ind],
                        spectra_data[0],
                        spectra_data[1])
print('the number of cropped data = {}'.format(interp_data.shape))

#def matching(spectra_data,minwave,maxwave,input_dim):
#    spectra_data[]

# padding size check, padarray gen., concat
min_ind = np.where(spectra_data[0]==minshift)
max_ind = np.where(spectra_data[0]==maxshift)
merged_array = interp_data
if min_ind[0].shape[0] == 0:
    min_ind = -1
    left_pad_size = int((spectra_data[0].iloc[0] - minshift)/unit_shift)
    print('left_pad_size : ',left_pad_size)
    left_array = np.zeros(left_pad_size)
    merged_array = np.concatenate((left_array,merged_array))
if max_ind[0].shape[0] == 0:
    max_ind = -1
    right_pad_size = int((maxshift - spectra_data[0].iloc[-1])/unit_shift)
    print('right_pad_size : ',right_pad_size)
    right_array = np.zeros(right_pad_size)
    merged_array = np.concatenate((merged_array,right_array))

print('after padding : {}',format(merged_array.shape))

merged_data_list = merged_array.tolist()

data=[{"spectra_data": merged_data_list,"pred": 0,"acc": 0,"probability": 0}]
r = requests.post('http://210.107.194.52:8000/predict',json=data)
print(r.status_code)
print(r.url)
print(r.text)
print(r.content)
print(r.encoding)
print(r.headers)