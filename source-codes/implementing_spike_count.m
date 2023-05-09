clear;
close all;
clc;

dir_expt = 'C:\Users\AAYUSHMAN\Sync\iBMI-Arindam-Basu\classification\monkey_1_set_1\expt\*.mat';
dir_train = 'C:\Users\AAYUSHMAN\Sync\iBMI-Arindam-Basu\classification\monkey_1_set_1\training\*.mat';
step = 0.1;
binWidth = 0.5;

[dataset,mean_f_rate,acc,pc] = spike_count(dir_expt,dir_train,step,binWidth,1);