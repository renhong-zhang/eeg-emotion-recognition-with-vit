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
import pandas as pd
import random
import glob
import pickle as pickle
from sklearn.preprocessing import normalize
import _pickle as cPickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import tqdm
from sklearn.model_selection import train_test_split
import fnmatch
import os
import random
import re
import threading
from six.moves import xrange
import time
import json
import torch as t
from torch.utils.data import Dataset,DataLoader
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

class Dataset_train(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, datapath,xwt,channel, rows,cols,val_arouse=0):
        'Initialization'
        self.channel = channel
        self.rows = rows
        self.cols = cols
        self.val_arouse = val_arouse
        data_filename = datapath+"/"+"final_data_train_"+xwt+".pkl"
        label_filename = datapath+"/"+"final_lables_train_"+xwt+".pkl"
        with open(data_filename,'rb') as filepath:
            self.new_dataset = pickle.load(filepath)
            filepath.close()
        with open(label_filename,'rb') as filepath:
            self.new_labels = pickle.load(filepath)
            filepath.close()
        self.new_dataset = self.new_dataset.reshape(-1, self.channel, self.rows, self.cols)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.new_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        
        # *Load data and get label

        X = self.new_dataset[index]

        y = self.new_labels[int(index / 32), self.val_arouse]

        if y >= 5:
            y = 1.0
        else:
            y = 0.0

        return X, np.array([y])
class Dataset_test(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, datapath,xwt,channel, rows,cols,val_arouse=0):
        'Initialization'
        self.channel = channel
        self.rows = rows
        self.cols = cols
        self.val_arouse = val_arouse
        data_filename = datapath+"/"+"final_data_test_"+xwt+".pkl"
        label_filename = datapath+"/"+"final_lables_test_"+xwt+".pkl"
        with open(data_filename,'rb') as filepath:
            self.new_dataset = pickle.load(filepath)
        with open(label_filename,'rb') as filepath:
            self.new_labels = pickle.load(filepath)
        self.new_dataset = self.new_dataset.reshape(-1, self.channel, self.rows, self.cols)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.new_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'

        # *Load data and get label
        X = self.new_dataset[index]

        y = self.new_labels[int(index / 32), self.val_arouse]

        if y >= 5:
            y = 1.0
        else:
            y = 0.0

        return X, np.array([y])