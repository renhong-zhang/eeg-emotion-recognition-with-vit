
import math
import six
import tensorflow as tf
from tensorflow import keras
import shutil
from enum import Enum
from einops.layers.tensorflow import Rearrange
import logging
import numpy as np
from fastprogress import master_bar, progress_bar
import random
import glob
import os
import pickle as pickle
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import fnmatch
import os
import random
import re
import threading
import numpy as npx
from six.moves import xrange
import time
import json
import torch as t
from torch.utils.data.distributed import DistributedSampler
from time import sleep
import math
import functools
from imblearn.over_sampling import RandomOverSampler
import absl.logging as _logging
import collections
import re
import six
from os.path import join
from six.moves import zip
from absl import flags
from sklearn.metrics import confusion_matrix,classification_report
from Model import*
from Dataset import*
from Trainer import*

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # *Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:  # *Overwrites any existing file.
        dataset_reload = pickle.load(input)

    return dataset_reload

def main():

    collate_fn = lambda batch: t.stack([t.from_numpy(b) for b in batch], dim=0)

    data_config = { 'path_':'/home/renhong_zhang/path-to-project/DEAP/test_out/my_erVITer_01/mydata',
                    'file_': 'cwt',
                    'chan_': 1,
                    'rows_':32,
                    'cols_':32,
                    'batch_':64,
                    'maxepochs_':100,
                    'patch_':8
                }
    
    chan_num=data_config['chan_']
    row_num=data_config['rows_']
    col_num=data_config['cols_']
    data_path=data_config['path_']
    xwt_file=data_config['file_']
    max_epochs=data_config['maxepochs_']
    batch_size=data_config['batch_']
    patch_size=data_config['patch_']
    
    dataset_train = Dataset_train(data_path,xwt_file,chan_num,row_num,col_num,0)
    dataset_test = Dataset_test(data_path,xwt_file,chan_num,row_num,col_num,0)

    print("Length of dataset is ", len(dataset_train))
    print("Length of dataset is ", len(dataset_test))

    train_loader = DataLoader(dataset_train, batch_size=data_config['batch_'], pin_memory=False, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=data_config['batch_'], pin_memory=False, shuffle=False)

    logger = logging.getLogger(__name__)

    tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=1e-5,chan_num=chan_num,row_num=row_num,col_num=col_num,ckpt_path=data_path)

    # *sample model config.
    model_config = {"image_size": row_num,
                    "patch_size": patch_size,
                    "num_classes": 2,
                    "dim": 512,
                    "depth": 6,
                    "heads": 8,
                    "mlp_dim": 1024,
                    "channels": chan_num,
                    "images": True
                    }

    trainer = Trainer(ViT, model_config, train_loader, len(dataset_train), test_loader, len(dataset_test), tconf)
    trainer.train()

if __name__ == "__main__":
    main()
