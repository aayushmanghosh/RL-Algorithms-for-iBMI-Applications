function spike_data = syn_dataset(directory,percentageNoise,neurons)
% SYN_DATASET Synthetic Neural Datasets
% 
%    syn_dataset(directory,percentageNoise,neurons) creates the synthetic
%    dataset based on the famous Izhikevich model for neural excitability. 
%    This code is a modified version of the Isekevich model of neural firing
%    pattern. This model is created to integrate the biologically palusible
%    Hodgkin-Huxley model and the computationally efficient integrate-and-
%    fire model. The model includes response from Excitatory and Inhibitory
%    Neurons to mimic the mammalian neuronal model. The ratio is placed at 4:1.
%
%    # Inputs:
%    - 'directory': The directory containing the classification files. The
%                   data is extracted using the code spike_count.m 
%    - 'percentageNoise': The input noise percentage.
%    - 'neurons': The number of neurons we want to use to generate our
%                 synthetic dataset.
%
%    # Outputs:
%    - 'spike_data': The number of spike counts according to the binWidth
%                    (0.5s) and step size (0.1s) across all the neurons 
%                    for all the sessions. The last column belongs to the 
%                    true labels (e.g - 1,2,3).
%
%   # Reference: Ghosh A., and Shaikh S. et al., Lightweight Reinforcement
%     Learning Decoders for Autonomous, Scalable, Neuromorphic
%     intra-cortical Brain Machine Interface; submitted Neuromorphic
%     Computing & Interface, 2023.
%
%   # Version: v1.0
%   # Context: This function is used to generate the synthetic dataset
%              based on the percentage noise level we prefer. The dataset
%              is further passed into the RL Algorithms to calculate the
%              decoding accuracy under various noise level. The dataset is
%              also used to calculate the variability.
%
% License: Please see the accompanying file named "LICENSE"
% Author: Aayushman Ghosh, University of Illinois Urbana Champaign, May 2023.
%         <aghosh14@illinois.edu>

% - Listing and loading the classification files.
folders = dir(directory);
targets = cell(length(folders), 1);
spike_data = cell(length(folders), 1);
for p = 1:length(folders)
    file = fullfile(folders(p).folder, folders(p).name);
    load(file);
    target = feature_mat(:,end)/90 + 1;
    targets{p,1} = target;
end

% - Preparing the exhibitory neuronal model.
Excite_neuron.Ne = neurons; % No. of neurons
Excite_neuron.re = rand(Excite_neuron.Ne, 1); % RV to bias the model towards RS neurons
Excite_neuron.a = 0.02*ones(Excite_neuron.Ne, 1); % equation param
Excite_neuron.b = 0.2*ones(Excite_neuron.Ne, 1); % equation param
Excite_neuron.c = -65 + 15*(Excite_neuron.re).^2; % equation param
Excite_neuron.d = 8 - 6*(Excite_neuron.re).^2; % equation param

% - Writing the synaptic weights and the corresponding mathematical models.
weights = Excite_neuron.Ne;
a = [Excite_neuron.a];
b = [Excite_neuron.b];
c = [Excite_neuron.c];
d = [Excite_neuron.d];
S = 0.5*rand(weights, Excite_neuron.Ne);
v = -65*ones(weights, 1); % Initialize
u = b.*v; % Initialize
saveFR = 1; % buffer

% - Creating the synthetic dataset
for i = 1:length(targets)
    final_target = targets{i,1};
    if saveFR
        str_stim = 5;
        str_nostim = 5;
        str_uncorr = 5;
        rng(1);
        ch_ind = randperm(weights); % Random permutation of synaptic weights.
        
        ch_fwd = ch_ind(1:abs(neurons/5));
        ch_right = ch_ind(abs(neurons/5)+1:2*abs(neurons/5));
        ch_left = ch_ind(2*abs(neurons/5)+1:3*abs(neurons/5));
        ch_stop = ch_ind(3*abs(neurons/5)+1:4*abs(neurons/5));
        ch_uncorr = ch_ind(4*abs(neurons/5)+1:5*abs(neurons/5));
        
        ch_all = [ch_right; ch_fwd; ch_left; ch_stop];
        rngInd = 1;
        firing_count_100ms = zeros(length(final_target), weights); % The firing count matrix.
        
        firing_ind = 1;
        for ind = 1:length(final_target)
            target_100ms = final_target(ind);
            I = zeros(weights,1); % create a stimulus input vector.
            I(ch_all(target_100ms,:)) = str_stim;
            stim_channels = ch_all(target_100ms,:);
            no_stim_channels = ch_all(setdiff(1:4,target_100ms),:);
            no_stim_channels = reshape(no_stim_channels,1,[]);
            Noise = 0.5 + percentageNoise;
            selectNoiseCh_nostim = randperm(length(no_stim_channels),round(Noise*length(no_stim_channels)));
            selectNoiseCh_stim = randperm(length(stim_channels),round(Noise*length(stim_channels)));
            selectNoiseCh = [no_stim_channels(selectNoiseCh_nostim) stim_channels(selectNoiseCh_stim)];
            I(selectNoiseCh) = str_nostim*randn(length(selectNoiseCh),1); % added noise to all no stim channels
            
            firings = [];
            for t = 1:100
                rng(rngInd);
                rngInd = rngInd + 1;
                I_uncorr = [str_uncorr*randn(length(ch_uncorr),1)]; % thalamic input
                I(ch_uncorr) = I_uncorr;
                
                fired = find(v>=30);
                firings = [firings; t+0*fired,fired];
                v(fired) = c(fired);
                u(fired) = u(fired) + d(fired);
                v = v + 0.5*(0.04*v.^2 + 5*v + 140 - u + I); % step 100 ms
                v = v + 0.5*(0.04*v.^2 + 5*v + 140 - u + I); % step 100 ms
                u = u + a.*(b.*v-u); % stability
            end
            firing_temp = zeros(1,weights);
            for ii = 1:weights
                firing_temp(ii) = sum(firings(:,2)==ii);
            end
            firing_count_100ms(firing_ind,:) = firing_temp;
            firing_ind = firing_ind + 1;
        end
        firing_count_500ms = firing_count_100ms;
        for ti = 5:size(firing_count_100ms,1)
            firing_count_500ms(ti,:) = sum(firing_count_100ms(ti-4:ti,:),1);
        end
    end
    reshape(final_target,[length(final_target) 1]);
    mask = mean(firing_count_500ms,1)>0;
    data = firing_count_500ms(:,mask);
    spike_data{i,1} = [data final_target]; % The final output vector.
end
end
