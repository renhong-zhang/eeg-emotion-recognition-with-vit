import os
import random
import re
import tensorflow as tf
import numpy as np
import time
import math
import _pickle as cPickle
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

sfreq = 128
rfreq = 512
fmin = 4
fmax = 45
baseline_secs = 3

init_args = {  'xwtdata_':'cwtdata',  #*rawdata,cwtdata,dwtdata,dtcwtdata
               'file_': 'cwt',        #*raw,cwt,dwt,dtcwt
               'chan_': 1,
               "rows_":32,
               'cols_':32,
               'freqs_':False,
               'wavesegs_':28,
               'src_':'/home/renhong_zhang/path-to-project/DEAP/test_out/my_erVITer_01',
               'dest_':'/home/renhong_zhang/path-to-project/DEAP/test_out/my_erVITer_01/mydata'
                }

wavesegments = init_args['wavesegs_']  
eeg_freqs = init_args['freqs_']   
directory = init_args['src_'] + '/' + init_args['xwtdata_']

def get_all_purp(root_dir):
    all_mat = []
    for dirName, _, fileList in os.walk(root_dir):
        for fname in fileList:
            if '.mat' in fname:
                all_mat.append(dirName + '/' + fname)
            elif '.bdf' in fname:
                all_mat.append(dirName + '/' + fname)
            elif '.dat' in fname:
                all_mat.append(dirName + '/' + fname)
    return all_mat
    
dataset = []
labels = []

for filename in get_all_purp(directory):
    print(filename)
    filedata=sio.loadmat(filename)
    if init_args['xwtdata_'] == 'rawdata':
        f_epoch=filedata[init_args['data']][:, :32, sfreq*baseline_secs:]
    else:
        f_epoch=filedata[init_args['xwtdata_']]
    y=filedata['labels'][:,0:2]
    """
    for vid in range(f_epoch.shape[0]):
        y[vid] =filedata['labels'][vid,0:2]
    """
    print('shape : ', len(f_epoch),' : ', f_epoch.shape, ' : ',)
    dataset.append(f_epoch)
    labels.append(y)

dataset = np.array(dataset)
labels = np.array(labels)
print('shape : ', len(dataset),' : ', dataset.shape)
print('shape : ', labels.shape)

if eeg_freqs == True:   #*alfa,beta,gama,...
    dataset = np.transpose(dataset,(0,1,3,4,2))
    new_dataset = np.zeros((32,40,wavesegments,5,9,9))
    Participant = 32
    Video = 40
    Channel = 32   
    for user in range(Participant):
        for tries in range(Video):
            for seg in range(wavesegments):
                for freq1 in range(5):
                    new_dataset[user,tries,seg,freq1][0]=(0,0,0,dataset[user,tries,seg,freq1,0],0,dataset[user,tries,seg,freq1,16],0,0,0)
                    new_dataset[user,tries,seg,freq1][1]=(0,0,0,dataset[user,tries,seg,freq1,1],0,dataset[user,tries,seg,freq1,17],0,0,0)
                    new_dataset[user,tries,seg,freq1][2]=(dataset[user,tries,seg,freq1,3],0,dataset[user,tries,seg,freq1,2],0,dataset[user,tries,seg,freq1,18],0,dataset[user,tries,seg,freq1,19],0,dataset[user,tries,seg,freq1,20])
                    new_dataset[user,tries,seg,freq1][3]=(0,dataset[user,tries,seg,freq1,4],0,dataset[user,tries,seg,freq1,5],0,dataset[user,tries,seg,freq1,22],0,dataset[user,tries,seg,freq1,21],0)
                    new_dataset[user,tries,seg,freq1][4]=(dataset[user,tries,seg,freq1,7],0,dataset[user,tries,seg,freq1,6],0,dataset[user,tries,seg,freq1,23],0,dataset[user,tries,seg,freq1,24],0,dataset[user,tries,seg,freq1,25])
                    new_dataset[user,tries,seg,freq1][5]=(0,dataset[user,tries,seg,freq1,8,],0,dataset[user,tries,seg,freq1,9],0,dataset[user,tries,seg,freq1,27],0,dataset[user,tries,seg,freq1,26],0)
                    new_dataset[user,tries,seg,freq1][6]=(dataset[user,tries,seg,freq1,11],0,dataset[user,tries,seg,freq1,10],0,dataset[user,tries,seg,freq1,15],0,dataset[user,tries,seg,freq1,28],0,dataset[user,tries,seg,freq1,29])
                    new_dataset[user,tries,seg,freq1][7]=(0,0,0,dataset[user,tries,seg,freq1,12],0,dataset[user,tries,seg,freq1,30],0,0,0)
                    new_dataset[user,tries,seg,freq1][8]=(0,0,0,dataset[user,tries,seg,freq1,13],0,dataset[user,tries,seg,freq1,31],0,0,0,)

    new_dataset = np.transpose(new_dataset,(0,1,2,4,5,3))
# *above for DE,PSD

all_labels = np.zeros((40*wavesegments*32,2))
   
count = 0
labelll = labels[:]
for sub in labelll:
    for label in sub:
        for i in range(wavesegments):
            all_labels[count] = label
            count += 1
            
if eeg_freqs == True: 
    new_dataset = new_dataset.reshape(-1,5,9,9)
else:
    dataset = dataset.reshape(-1, 32, init_args['rows_'], init_args['cols_'], 1)
    dataset = np.transpose(dataset,(0,1,4,2,3))    
all_labels=all_labels.reshape(-1,2)

traindatafile = init_args['dest_'] + '/final_data_train_' + init_args['file_'] + '.pkl'
trainlabelsfile = init_args['dest_'] + '/final_lables_train_' + init_args['file_'] + '.pkl'
testdatafile = init_args['dest_'] + '/final_data_test_' + init_args['file_'] + '.pkl'
testlabelsfile = init_args['dest_'] + '/final_lables_test_' + init_args['file_'] + '.pkl'

X_train, X_test, y_train, y_test = train_test_split(dataset, all_labels, test_size=0.4, random_state=42)

"""
#std normalize
X_train_2d = X_train.reshape(-1, init_args['cols_'])
X_test_2d = X_test.reshape(-1, init_args['cols_'])
stdScale = preprocessing.StandardScaler().fit(X_train_2d)
X_train_2d = stdScale.transform(X_train_2d)
X_test_2d = stdScale.transform(X_test_2d)
X_train = X_train_2d.reshape(-1, 32, init_args['rows_'], init_args['cols_'], 1)
X_test = X_test_2d.reshape(-1, 32, init_args['rows_'], init_args['cols_'], 1)
##
"""
print(f"Train Data: {len(X_train)}")
print(f"Validation Data: {len(X_test)}")
with open(traindatafile, 'wb') as filepath:
      cPickle.dump(X_train, filepath)
with open(trainlabelsfile, 'wb') as filepath:
      cPickle.dump(y_train, filepath)
with open(testdatafile, 'wb') as filepath:
      cPickle.dump(X_test, filepath)
with open(testlabelsfile, 'wb') as filepath:
      cPickle.dump(y_test, filepath)