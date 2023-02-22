from __future__ import print_function

from itertools import chain
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import pickle as pickle
from vit_pytorch import ViT
GPU_NUM = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Available devices are',device)
if device == 'cuda':
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))

data_config = { 'path_':'/home/renhong_zhang/path-to-project/DEAP/test_out/my_erVITer_01/mydata',
                'file_': 'cwt',
                'chan_': 1,
                'rows_':32,
                'cols_':32,
                'batch_':512,
                'patch_':8,
                'maxepochs_':100,
                'freq_image_':False,
                'freqs_':False
                }
    
chan_num=data_config['chan_']
row_num=data_config['rows_']
col_num=data_config['cols_']
data_path=data_config['path_']
xwt_file=data_config['file_']
max_epochs=data_config['maxepochs_']
batch_size=data_config['batch_']
patch_size=data_config['patch_']
freq_image_mode=data_config['freq_image_']
freqs = data_config['freqs_']

# *Training settings
epochs = max_epochs
lr = 0.00027869504431781426
gamma = 0.9
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = False

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=(0.2, 2), 
                               contrast=(0.3, 2), 
                               saturation=(0.2, 2), 
                               hue=(-0.3, 0.3)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.225, 0.225, 0.225)),
    ]
)
        
val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.225, 0.225, 0.225)),
    ]
)
        
seed_everything(seed)

class eegDataset(Dataset):
    def __init__(self,datapath,xwt,channel, rows,cols,val_arouse=0,freq_image_mode=False,freqs_mode=False,test=False):
        'Initialization'
        self.channel = channel
        self.rows = rows
        self.cols = cols
        self.valence_arouse = val_arouse
        self.freq_image_mode = freq_image_mode
        self.freqs_mode = freqs_mode
        self.test =test
        
        if self.test == False:
            data_filename = datapath+"/"+"final_data_train_"+xwt+".pkl"
            label_filename = datapath+"/"+"final_lables_train_"+xwt+".pkl"
        else:
            data_filename = datapath+"/"+"final_data_test_"+xwt+".pkl"
            label_filename = datapath+"/"+"final_lables_test_"+xwt+".pkl"
            
        with open(data_filename,'rb') as filepath:
            self.new_dataset = pickle.load(filepath)
            filepath.close()
        with open(label_filename,'rb') as filepath:
            self.new_labels = pickle.load(filepath)
            filepath.close()

        if self.freq_image_mode == False:
            _,chans,rgbs,rows,cols=self.new_dataset.shape    
            self.new_dataset = self.new_dataset.reshape((-1,rgbs,rows,cols))
            if self.freqs_mode ==False:
                self.chans = chans
            else:
                self.chans = 1      #*9*9 spatial array of channel
        else:
            nums,freqs,chans,wavesegs=self.new_dataset.shape
            if chans>wavesegs:
                self.new_dataset.resize((nums,freqs,chans,chans),refcheck=False)
            else:
                if chans<wanvesegs:
                    self.new_dataset.resize((nums,freqs,wavesegs,wavesegs),refcheck=False)
            self.chans = 1   #*channel &timeserise dot data is constructed an big image
    
    def __len__(self):
        return len(self.new_dataset)

    def __getitem__(self, index):
        image_array = self.new_dataset[index]
        label = self.new_labels[int(index/self.chans),self.valence_arouse]
        if label >= 5:
          label = 1
        else:
          label = 0
         
        return image_array, label

train_data = eegDataset(data_path,xwt_file,chan_num,row_num,col_num,0,freq_image_mode,freqs,False)
valid_data = eegDataset(data_path,xwt_file,chan_num,row_num,col_num,0,freq_image_mode,freqs,True)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

'''================================================================
Total params: 271,590,402
Trainable params: 271,590,402
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 1038.92
Params size (MB): 1036.04
Estimated Total Size (MB): 2075.53
---------------------------------------------------------------- '''

model = ViT(
    dim=128,        #*1024,---128
    image_size=row_num,  #*32 cwt
    patch_size=patch_size, #*16
    num_classes=2,
    depth= 32, # *number of transformer blocks 32---8
    heads= 16, # *number of multi-channel attention---8
    mlp_dim=128,   #*1024---128
    channels=chan_num,     #*freqs
)

model.to(device)

# *loss function
criterion = nn.CrossEntropyLoss()
# *optimizer
optimizer = optim.RMSprop(model.parameters(), lr=lr)
# *scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    model.train()
    for data, label in tqdm(train_loader):
        data = data.to(torch.float32).to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        """
        loss = criterion(outputs, labels)
        flood = (loss-b).abs()+b # not let loss is too small set b equal to a fit value
        optimizer.zero_grad()
        flood.backward()
        optimizer.step()
        """
        """
        regularization_loss = 0
        for param in model.parameters():
            if param.requires_grad:
                regularization_loss += torch.sum(torch.abs(param))
        classify_loss = criterion(output, label)
        loss = classify_loss+0.00001 * regularization_loss
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(torch.float32).to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
    scheduler.step()
