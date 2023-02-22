%% ~clear all/ close figs
close all
clear
clc

%% ~default paramters
Participant = 32;
Video = 40;
Channel = 32;
wavesegments = 10

addpath('/home/renhong_zhang/path-to-project/DEAP/data_preprocessed_matlab')

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
    framesize=60//wavesegments*128;
    cwtdata = zeros(Video,Channel, wavesegments, totalScale, totalScale,1);    

    for video=1:Video
        
        fprintf('\ncreating file participant %d,video %d:\n',participant,video);
 
        for channel = 1:Channel
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
                
                %*decompose into wavelets
                [c,l] = dualtree(data2,'Level',3,'FilterLength',14);
                coefs = l{3}
                dwtdata(video,channel, 10, :, :,1) = coefs;
            end %*wavesegmentloop
        end %*channel
         
    end %*the testcase loop
    save(filename,'dwtdata');
    save(filename,'labels','-append');
end %*the file loop