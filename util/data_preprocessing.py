import requests
import json
import numpy as np
import pandas as pd
import boto3

MINSHIFT = 382
MAXSHIFT = 1792
INPUTDIM = 1000

def raman_shift_matching(spectra_data):

    minshift = MINSHIFT
    maxshift = MAXSHIFT
    input_dim = INPUTDIM

    unit_shift = (maxshift - minshift)/input_dim
    print("unit_shift = ",unit_shift)

    raman_shift_arr = np.array(spectra_data['raman_shift'])
    intensity_arr = np.array(spectra_data['intensity'])

    # given spectrum data interpolation for NN input format
    if minshift <= raman_shift_arr[0]:
        left_pad_flag=1
    else:
        left_pad_flag=0
    if raman_shift_arr[-1] <= maxshift:
        right_pad_flag=1
    else:
        right_pad_flag=0

    interp_shift = np.arange(minshift,maxshift+0.0000001,unit_shift)

    print(interp_shift.shape,interp_shift[0],interp_shift[-1])
    if left_pad_flag :
        interp_min_ind = np.where((interp_shift>raman_shift_arr[0] -unit_shift/2) &
                (interp_shift<raman_shift_arr[0] +unit_shift/2))[0][0]
    else :
        interp_min_ind = np.where((interp_shift>minshift -unit_shift/2) &
                (interp_shift<minshift +unit_shift/2))[0][0]
    if right_pad_flag:
        interp_max_ind = np.where((interp_shift>raman_shift_arr[-1] -unit_shift/2) &
                (interp_shift<raman_shift_arr[-1] +unit_shift/2))[0][0]
    else :
        interp_max_ind = np.where((interp_shift>maxshift -unit_shift/2) &
                (interp_shift<maxshift +unit_shift/2))[0][0]

    print('given data - left : [{}]={}, given data - right : [{}]={}'
            .format(interp_min_ind,interp_shift[interp_min_ind],interp_max_ind,interp_shift[interp_max_ind]))

    interp_data = np.interp(interp_shift[interp_min_ind:interp_max_ind],
            raman_shift_arr,
            intensity_arr)

    print('the number of cropped data = {}'.format(interp_data.shape))


    #def matching(spectra_data,minwave,maxwave,input_dim):
    #    spectra_data[]

    # padding size check, padarray gen., concat
    min_ind = np.where(raman_shift_arr==minshift)
    max_ind = np.where(raman_shift_arr==maxshift)
    merged_array = interp_data
    if min_ind[0].shape[0] == 0:
        min_ind = -1
        left_pad_size = int((raman_shift_arr[0] - minshift)/unit_shift)
        print('left_pad_size : ',left_pad_size)
        left_array = np.zeros(left_pad_size)
        merged_array = np.concatenate((left_array,merged_array))
    if max_ind[0].shape[0] == 0:
        max_ind = -1
        right_pad_size = int((maxshift - raman_shift_arr[-1])/unit_shift)
        print('right_pad_size : ',right_pad_size)
        right_array = np.zeros(right_pad_size)
        merged_array = np.concatenate((merged_array,right_array))

    print('after padding : {}'.format(merged_array.shape))

    merged_data_list = merged_array.tolist()

    return merged_data_list
