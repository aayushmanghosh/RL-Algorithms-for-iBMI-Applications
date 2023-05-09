function [dataset,mean_f_rate,acc,pc] = spike_count(dir_expt,dir_train,step,binWidth,v)
%function [dataset,mean_f_rate] = spike_count(dir_expt,dir_train,step,binWidth,v)
%
% SPIKE_COUNT Counts the number of spikes occuring in a bin. The code has
% two options. The first one integartes an LDA (Linear Discriminant
% Algorithm), and reports its accuracies across experimental data. It also
% integrates the Principal Component Analysis, and reports the first two
% PCs (Principal Components).
%
%   SPIKE_COUNT counts the number of spikes occuring in a backward looking
%   bin of 0.5s, that is moving at a step of 0.1s throughout the whole time
%   period of the experiment. The code is not autonomous, it works only for
%   the datasets that are presented and recorded in a particular manner. It
%   works especially for the datasets recorded for this study. Please see
%   context for more information. The code also offers a choice to see the
%   accuracy of the extracted dataset using the LDA Algorithm, and to
%   collect the Principal Components.
%
%   This code is a publication specific script but the Binwidth (here, 0.5s
%   ), and the step (here, 0.1s) can be adjusted as per requirement.
%
%   # Inputs:
%   - 'dir_expt': The directory containing the experiment files.
%   - 'dir_train': The directory containing the training files.
%   - 'step': The step size for the Algorithm.
%   - 'binWidth': The bin size of the backward looking window.
%   - 'v': Logical choice (either 0 or 1) to enable plots or not.
%
%   # Outputs:
%   - 'dataset': The number of spike counts according to the binWidth
%                and step size across all the electrodes for all the sessions. 
%                The last column belongs to the true labels (e.g - 0, 90, 180).
%
%   - 'mean_f_rate': Mean across each columns for all the electrodes.
%   - 'acc': The accuracy across all the datasets using LDA.
%   - 'pc': First two principal components of each dataset.
%
%   # Reference: Ghosh A., and Shaikh S. et al., Lightweight Reinforcement
%     Learning Decoders for Autonomous, Scalable, Neuromorphic
%     intra-cortical Brain Machine Interface; submitted Neuromorphic
%     Computing & Interface, 2023.
%
%   # Version: v1.0
%   # Context: This function is used to generate the training and testing
%              data for calculating the decoding accuracy of the RL
%              Algorithms. This code is tailored specifically for the open
%              access dataset that is available at <https://osf.io/dce96/>
%
% License: Please see the accompanying file named "LICENSE"
% Author: Aayushman Ghosh, University of Illinois Urbana Champaign, May 2023.
%         <aghosh14@illinois.edu>

% - Listing the files in the input experiment and training folders.
folders_expt = dir(dir_expt);
folders_train = dir(dir_train);

% - Defining the empty cells to record the accuracies and the PCs.
acc = zeros(length(folders_expt), 1);
pc = cell(length(folders_expt),1);
dataset = cell(length(folders_expt),1);

% - Reading each files to determine the sent signals (0, 90, 180) and how
%   many electrodes are used to record the data for each session.
for p = 1:length(folders_expt)
    file_expt = fullfile(folders_expt(p).folder, folders_expt(p).name);
    load(file_expt);
    file_train = fullfile(folders_train(p).folder, folders_train(p).name);
    load(file_train);
    
    % There is an irregularity in the database. Different variable name is
    % used to determine the choice of the sent signals (0, 90, 180). The
    % following lines compare between both the choices to see which one is
    % present, and extract those commands.
    choice = ["SentSignals" "AllSentCommands"];
    Variables = fieldnames(IMETrainingData);
    index = find(not(~contains(Variables,choice(2))), 1);
    if (isempty(index) == 1)
        commands = IMETrainingData.SentSignals;
    else
        commands = IMETrainingData.AllSentCommands;
    end
    
    date = zeros(length(IMETrainingData.Timestamps), 3); % To determine the date and timestamp.
    features = cell(length(commands),1); % - Define a cell to hold the features (no. of spikes). 
    for i = 1:length(commands)
        tstart = IMETrainingData.StartTime(i);
        tend = IMETrainingData.EndTime(i);
        binEdges = tstart:step:tend; % The forward time step
        sent_sigs = commands{i,1};
        spikes = zeros(min([numel(binEdges) numel(sent_sigs)]), size(Spike_data,1));
        date(i,:) = [IMETrainingData.Timestamps{i,1}(1,3) IMETrainingData.Timestamps{i,1}(1,2) IMETrainingData.Timestamps{i,1}(1,1)];
        
        % Calculating the number of spikes in a backward looking window. 
        for j=1:size(spikes,1)
            for k=2:size(spikes,2)
                spikes(j,k-1) = sum((Spike_data{k,3}>=binEdges(j)-binWidth) & (Spike_data{k,3}<binEdges(j)));
            end
            spikes(j,end) = sent_sigs(j);
            spikes(any(isnan(spikes), 2), :) = [];
            spikes(all(~spikes,2), :) = [];
        end
        % Writing the features, and erasing the empty cells.
        features{i,1} = spikes;
        features = features(~cellfun('isempty', features'));
    end
    
    % Converting from cell to .mat format for easy integration with further
    % steps (using as a dataset for RL Algorithms).
    feature_mat = cell2mat(features);
    mean_f_rate = mean(feature_mat(:,1:end-1),1);
    feature_mat = feature_mat(:,[mean_f_rate>=2 true]);
    dataset{p,1} = feature_mat;
    
    % Writing the LDA Algorithm.
    X = zscore(feature_mat(:,1:end-1),[],1); % Spike count data.
    y = feature_mat(:,end); % True labels.
    trainSize = floor(size(feature_mat,1)*0.6); % 60% is used as the training dataset
    [class,~,~,~,~] = classify(X(trainSize+1:end,:), X(1:trainSize,:), y(1:trainSize)); % The LDA Algorithm
    true_class = feature_mat(trainSize+1:end,end); % True labels.
    pred_class = class; % Predicted labels.
    accuracy = sum(true_class == pred_class)/size(true_class, 1)*100; % Determing the accuracy.
    acc(p,1) = accuracy;
    
    % Calculating the Principal Components for the feature set.
    [coeff, ~, ~, ~, explained] = pca(feature_mat(:,1:end-1)');
    pcs = [coeff(:,1) coeff(:,2) y];
    pc{p,1} = pcs;
    
    % Choice to see the plots for the Principal Components.
    if v == 1
        subplot(4,2,p);
        gscatter(pcs(:,1),pcs(:,2),pcs(:,3),[],'^');
        ylabel('PC1');
        xlabel('PC2');
        title(sprintf('Classification accuracy: %.2f \n Total Variance: %.2f',acc(p,1),(explained(1,1) + explained(2,1))));
    end
end

% Choice to see the accuracy from the LDA with the progression of the
% sessions in a dataset.
if v == 1
    figure(2)
    plot(acc, '--bo', 'LineWidth', 2);
    title('Classification Accuracy over various sessions');
    xlabel('Progression of Sessions');
    ylabel('Classification Accuracy');
    grid on;
end
end

