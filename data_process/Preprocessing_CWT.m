%% ~clear all/ close figs
close all
clear
clc

%% *default paramters
Participant = 32;
Video = 40;
Channel = 32;
wavesegments = 10
Fs = 128;
Time = 63;
addpath('/home/renhong_zhang/path-to-project/DEAP/data_preprocessed_matlab')

%% ~set parameters
totalScale = 32;
wname = 'coif5';

for participant = 1:Participant
    if(participant<10)
        myfilename = sprintf('s0%d.mat', participant);
    else
        myfilename = sprintf('s%d.mat', participant);
    end
    load(myfilename);
    filename = ['/home/renhong_zhang/path-to-project/cwtdata/cwt_' myfilename]
              
    datastart=128*3;
    datalength=8064-datastart;
    framesize=60/wavesegments*128;
    cwtdata = zeros(Video,Channel, wavesegments, totalScale, totalScale,1);      
    
    for video=1:Video
        
        fprintf('\ncreating file participant %d,video %d:\n',participant,video);
  
        for channel = 1:Channel
            """
            for wavesegmentloop = 1:wavesegments
                data1=zeros(1,framesize);
                dataa=framesize*wavesegmentloop;
                iii = 1;
                for ii = dataa-framesize+1:dataa
                    data1(1,iii)=data(video,channel,ii+datastart);
                    iii = iii + 1;
                end
            """
            data1=zeros(1,8064-datastart);
            for ii =1:datalength
                data1(1,ii)=data(video,channel,ii+datastart);
            end
            start=0;
            %*take the wavesegment from the main wave
            for wavesegmentloop=1:wavesegments
                
                data2=zeros(1,framesize);
                for jj =1:framesize
                    data2(1,jj)=data1(1,start+jj);
                end
                start=start+framesize;
   
                % *decompose into wavelets
                % *set scales
                
                f = 1:totalScale;
                f = Fs/totalScale * f;
                wcf = centfrq(wname);
                scal =  Fs * wcf./ f;           
                coefs = cwt(data2, scal, wname);
                
                coefs = imresize(coefs, [totalScale, totalScale]);
            
                cwtdata(video,channel, wavesegmentloop, :, :,1) = coefs;
            
            end %*wavesegmentloop
        end %*channel

    end %*video loop
    save(filename,'cwtdata');
    save(filename,'labels','-append');
end %*participant loop