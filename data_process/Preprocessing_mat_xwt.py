import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle as pickle
import _pickle as cPickle
import antropy as ant
from scipy.stats import entropy
import scipy.io as sio
import dtcwt
import pywt

def TimeFrequencyCWT(data,sampling_rate,total_scale,wavelet='morl'):
    
    # *center frequency
    wcf = pywt.central_frequency(wavelet=wavelet)
    #*wavelet scaler
    cparam = 2 * wcf * total_scale
    scales = cparam/np.arange(total_scale, 1, -1)
    # *continuous wavelet transform
    [cwt_matrix, frequencies] = pywt.cwt(data, scales, wavelet, 1.0/sampling_rate)
    return [cwt_matrix, frequencies]

def feature_cnt(fq_d):
    ma=np.mean(fq_d)
    if ma >0 :
        mavx=np.log(ma)
    else:
        mavx=0
    psdx=np.mean(fq_d*np.conj(fq_d))/len(fq_d)
    dfax=ant.higuchi_fd(np.squeeze(np.real(fq_d)))
                
    pd_series=pd.Series(np.squeeze(np.real(fq_d)))
    entropyx=entropy(pd_series.value_counts())
    return np.real(mavx),np.real(psdx),dfax,entropyx

init_args = {  'xwtdata_':'cwtdata',  #*rawdata,cwtdata,dwtdata,dtcwtdata
               'file_': 'cwt_',        #*cwt,dwt,dtcwt
               'chan_': 1,
               "rows_":32,               #*5,4  /32,32  /plus 1
               'cols_':32,
               'freqs_':False,          #*eeg frequency
               'src_':'',
               'dest_':''
                }

#*default paramters
Participant = 32
Video = 40
Channel = 32
freqs = 128

#*cwt
row_total_scale = init_args['rows_']+1; #*cwt
col_total_scale = init_args['cols_']+1
wname = 'morl'       #*malab is 'coif5';
eeg_freqs = init_args['freqs_']

windowsize = 6  #*15s,12s,10s,9s,8s,6s,4s,3s,2s
overlapsize= 4  # *1s,2s,3s,4s,5s,6s,7s,8s,9s,10s
stepsize = windowsize - overlapsize #*1s,2s,3s,4s,5s,6s

if (60 - windowsize)%(windowsize-overlapsize) != 0 :
    print('windowsize/overlapsize err\n')
    quit()

wave_segments =(60 - windowsize)//(windowsize-overlapsize)+1

#*dtcwt function
if init_args['xwtdata_']=='dtcwtdata':
    transform = dtcwt.Transform1d()

path1=init_args['src_']
path2=init_args['dest_'] + init_args['xwtdata_'] + '/'
print(init_args['xwtdata_'])

for participant in range(Participant):
    if(participant<9):
        name ='%0*d' % (2,participant+1)
    else:
        name = participant+1
    
    my_filename = 's'+str(name)+'.mat'
    
    raw_filename=path1+my_filename 
    filedata = sio.loadmat(raw_filename)
    data = filedata['data']
    labels = filedata['labels']
    dt_filename =path2 + init_args['file_'] + my_filename
           
    data_start=freqs*3
    data_length=8064-data_start
    frame_size=windowsize*freqs
    
    xwt_data =np.zeros((Video,Channel, wave_segments,row_total_scale-1,col_total_scale-1)) 
        
    for video in range(Video):
        
        print("\ncreating file participant:%d video :%d\n"%(participant,video))
        
        for channel in range(Channel):
            data1=np.zeros((1,8064-data_start))
            for ii in range(data_length):
                data1[0,ii]=data[video,channel,ii+data_start]
            start=0
            #*take the wavesegment from the main wave
            for wave_segment_loop in range(wave_segments):
                
                data2=np.zeros((1,frame_size))
                fmatrix=np.zeros((row_total_scale-1,col_total_scale-1))
                for jj in range(frame_size):
                    data2[0,jj]=data1[0,start+jj]
                start=start+stepsize*freqs
                
                if init_args['xwtdata_']=='dtcwtdata':
                    #*decompose into wavelets
                    data2=data2.reshape(-1,1)
                    vecs_t = transform.forward(data2, nlevels=5)
                    delta=vecs_t.highpasses[0]
                    theta=vecs_t.highpasses[1]
                    alpha=vecs_t.highpasses[2]
                    beta=vecs_t.highpasses[3]
                    gamma=vecs_t.highpasses[4]
                    if eeg_freqs == True:
                        data_dtcwt_list=[delta,theta,alpha,beta,gamma]
                        for (i,data_) in enumerate(data_dtcwt_list):
                            fmatrix[i,0],fmatrix[i,1],fmatrix[i,2],fmatrix[i,3]=feature_cnt(data_)
                    else:
                        aa=np.real(delta)[0:32*6]
                        aa=np.concatenate((aa, np.imag(delta)[0:32*6]), axis=0)
                        aa=np.concatenate((aa, np.real(theta)[0:32*6]), axis=0)
                        aa=np.concatenate((aa, np.imag(theta)[0:32*6]), axis=0)
                        aa=np.concatenate((aa, np.real(alpha)[0:32*3]), axis=0)
                        aa=np.concatenate((aa, np.imag(alpha)[0:32*3]), axis=0)
                        aa=np.concatenate((aa, np.real(beta)[0:32*1]), axis=0)
                        aa=np.concatenate((aa, np.real(beta)[0:32*1]), axis=0)
                        aa=aa.reshape(row_total_scale-1,col_total_scale-1)   #*must be 32,32
                        fmatrix[:,:]=aa[:,:]
                elif init_args['xwtdata_']=='dwtdata':
                    coeffs = pywt.wavedec(data2,'db4',level=5)
                    cA5, cD5,cD4,cD3,cD2,cD1 = coeffs
                
                    delta=cD1   #*row 0
                    theta=cD2   #*row 1
                    alpha=cD3   #*row 2
                    beta=cD4   #*row 3
                    gamma=cD5   #*row 4
                    if eeg_freqs == True:
                        data_dtcwt_list=[delta,theta,alpha,beta,gamma]
                        for (i,data_) in enumerate(data_dtcwt_list):
                            fmatrix[i,0],fmatrix[i,1],fmatrix[i,2],fmatrix[i,3]=feature_cnt(data_)
                    else:
                        aa=data2[0,:254].reshape(-1)
                    for i in range(1,6):
                        bb=coeffs[i]
                        bb=bb.reshape(-1)
                        aa=np.concatenate((aa, bb), axis=0)
                        aa=aa.reshape(row_total_scale-1,col_total_scale-1)  #*must be 32,32
                        fmatrix[:,:]=aa[:,:] #*no use only can be understand
                elif init_args['xwtdata_']=='cwtdata':
                    [fmatrix,frequency]=TimeFrequencyCWT(data2[0], sampling_rate = freqs, total_scale=row_total_scale, wavelet=wname)
                    fmatrix=fmatrix.reshape(row_total_scale-1,-1)
                    fmatrix=fmatrix[:,0:row_total_scale-1]     #*can be any,any
                else:
                    quit()
                        
                xwt_data[video,channel, wave_segment_loop, :, :] = fmatrix[:,:] #*,psd,,de
                
    #*the testcase loop
    xwt_data=xwt_data.tolist()
    labels=labels.tolist()
    sio.savemat(dt_filename,{init_args['xwtdata_']:xwt_data,"labels":labels})
   
#*the file loop