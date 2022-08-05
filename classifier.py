from time import time, localtime, strftime
import numpy as np
import pandas as pd
import boto3

from resnet import ResNet
import os
import torch

from training import get_predictions
from datasets import spectral_dataloader

from util.data_preprocessing import raman_shift_matching

from data_structure.input_structure import BatchStructure

def classify(batch_dic):
    results = {}
    for k, batch in batch_dic.items():
        t00 = time()

        #BUCKET_NAME = 'ramcell-test-spectra-source'
        #OBJECT_NAME = 'Glucose 500mW 10sec.txt'
        #OBJECT_NAME = 'Glucose 500mW 10sec.csv'
        #OBJECT_NAME = config.spectra[i]

        #ext = OBJECT_NAME.split('.')[1]
        #print("Object data type : {}".format(ext))
        #s3 = boto3.client('s3')
        #target_data_path = './download/inference_target_data.'+ext
        #s3.download_file(BUCKET_NAME, OBJECT_NAME,target_data_path)

        #if ext == 'txt' : data = pd.read_csv(target_data_path, sep="\t", header=None)
        #else : data = pd.read_csv(target_data_path, header=None)


        #X_axis = np.arange(0,1000)
        #data_npy = data[1].to_numpy()
        #input_data = np.interp(X_axis,np.linspace(0,1000,num=data_npy.shape[0]),data_npy)
        #X = input_data.reshape(1,1000)
        X_batch = []
        for spectrum_ind, intensity in enumerate(batch.intensity):
            spectrum_data = {'intensity':intensity, 'raman_shift':batch.raman_shift[spectrum_ind]}
            X = raman_shift_matching(spectrum_data)
            X_batch.append(np.array(X))

        X_batch = np.vstack(X_batch)
        print("X_batch shape = {}".format(X_batch.shape))

        # CNN parameters
        layers = 6
        hidden_size = 100
        block_size = 2
        hidden_sizes = [hidden_size] * layers
        num_blocks = [block_size] * layers
        input_dim = 1000
        in_channels = 64
        n_classes = 30

        GPU_NUM = batch.gpu_id # 원하는 GPU 번호 입력
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device) # change allocation of current GPU
        print ('Current cuda device ', torch.cuda.current_device()) # check
        if device != 'cpu' : cuda = True

        # Load trained weights for demo
        cnn = ResNet(hidden_sizes, num_blocks, input_dim=input_dim,
                        in_channels=in_channels, n_classes=n_classes)
        cnn.cuda()
        cnn.load_state_dict(torch.load(
            batch.model, map_location=lambda storage, loc: storage))

        #y = np.array([0])
        y = np.array(batch.label)

        dl_test = spectral_dataloader(X_batch, y, batch_size=batch.batch_size,num_workers=1)
        probs_list, preds, acc = get_predictions(cnn,dl_test,cuda)
        # prob_test= get_predictions(cnn,dl_test,cuda,get_probs=True)

        print(preds, acc)
        print(probs_list)
        results[batch.model] = [preds, acc, probs_list] # [List[int], float, List[List[float]]]
    return results
