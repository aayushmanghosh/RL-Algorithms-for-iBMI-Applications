function RL_plots(expt_no,days,order,varargin)
% RL_PLOTS -- Performance plots for Reinforcement Learning based iBMI Decoders.
%
%    RL_plots provides the Decoding accuracy across various experiments for
%    different Reinforcement Learning Algorithms, and supervised learning
%    algorithms (for ideal cases, where there is no error or sparsity with
%    feedback signals). The code allows autonomous control of choosing
%    between different cases, and generates publication-specific plots for
%    each case.
%
%    This code is a publication specific script but can be adjusted as per
%    requirements. Please follow the accompanying comments.
%
%    # Inputs:
%    - 'expt_no': defines the experiment name and number (string input).
%                 For example: expt_no = "Experiment 1"; "Experiment 2"; 
%                 "Experiment 3"; ...
%    - 'days': assignes the used x axis.
%              For example: [1 5 12 19 26 40 47 64]; [1 10 17 29 35 53 56
%              63]; [1 4 6 62]; [1 10 35 79]; ...
%    - 'order': defines the order of input. The accepted inputs are of the
%               following types:
%               (1) order = ["Mean_HRL" "stdHRL" "Mean_AGREL" "stdAGREL" "Mean_Q"...
%                   "stdQ" "Mean_Band" "stdBand" "Mean_BandRP" "stdBandRP"];
%               (2) order = = ["LDA_retrain" "SVM_retrain" "LDA_fixed" "SVM_fixed" ...
%                   "Mean_HRL" "stdHRL" "Mean_AGREL" "stdAGREL" "Mean_Q"...
%                   "stdQ" "Mean_Band" "stdBand" "Mean_BandRP" "stdBandRP"];
%                 
%    - 'varargin': defines the input data. It has to be in a row vector
%       format and based on the order, where Mean_HRL means the mean of the
%       HRL Algorithm, and stdHRL means the standard deviation of the HRL
%       Algorithm. This order follows on:
%
%    # Reference: Ghosh A., and Shaikh S. et al., Lightweight Reinforcement
%    Learning Decoders for Autonomous, Scalable, Neuromorphic
%    intra-cortical Brain Machine Interface; submitted Neuromorphic
%    Computing & Interface, 2023.
%
%    # Version: v1.0
%    # Context: This function is used to generate the publication specific
%               plots. The accompanying scripts generates the plots. <refer: Fig. 4,
%               5, 7, 8, and 9 of the main text; and S1, S4, S5 of the
%               supplementary text.
%
% License: Please see the accompanying file named "LICENSE"
% Author: Aayushman Ghosh, University of Illinois Urbana Champaign, May 2023.
%         <aghosh14@illinois.edu>

% - Switching cases to see whether we have to plot for the ideal case or
% the cases with error and sparsity in feedback signal.
switch nargin
    case 10 % for errors and sparsity in feedback signal.
        x = input('Have you entered the Input in the following order -- "Mean_Alg" "stdAlg" (yes/no)?','s');
        if strcmpi(x,'yes')
            match = ["Mean_HRL" "stdHRL" "Mean_AGREL" "stdAGREL" "Mean_Q"...
                "stdQ" "Mean_Band" "stdBand" "Mean_BandRP" "stdBandRP"];
            proper = strcmpi(order,match);
            if sum(proper) == 10 % check to determine whether the exact order is given.
                disp("Input arguments passed:" + nargin + ", and they are in correct order");
                Bacon_HRL_mean = varargin{1};
                Bacon_HRL_std = varargin{2};
                Bacon_AGREL_mean = varargin{3};
                Bacon_AGREL_std = varargin{4};
                Bacon_Q_mean = varargin{5};
                Bacon_Q_std = varargin{6};
                Bacon_Band_mean = varargin{7};
                Bacon_Band_std = varargin{8};
                Bacon_BandRP_mean = varargin{9};
                Bacon_BandRP_std = varargin{10};
            end
        elseif strcmpi(x, 'no')
            disp('Unexpected inputs - Please enter all the arguments in proper order');
        else
            error('Please type "yes" or "no"');
        end
    case 12 % for ideal cases, includes the supervised algorithms.
        x = input('Have you entered the Input in the following order -- "Mean_Alg" "stdAlg" (yes/no)?','s');
        if strcmpi(x,'yes')
            match = ["LDA_retrain" "SVM_retrain" "LDA_fixed" "SVM_fixed" ...
                "Mean_HRL" "stdHRL" "Mean_AGREL" "stdAGREL" "Mean_Q"...
                "stdQ" "Mean_Band" "stdBand" "Mean_BandRP" "stdBandRP"];
            proper = strcmpi(order,match);
            if sum(proper) == 14 % check to see whether the exact order is given.
                disp("Input arguments passed:" + nargin + ", and they are in correct order");
                LDA_retrain = varargin{1};
                SVM_retrain = varargin{2};
                LDA_fixed = varargin{3};
                SVM_fixed = varargin{4};
                Bacon_HRL_mean = varargin{5};
                Bacon_HRL_std = varargin{6};
                Bacon_AGREL_mean = varargin{7};
                Bacon_AGREL_std = varargin{8};
                Bacon_Q_mean = varargin{9};
                Bacon_Q_std = varargin{10};
                Bacon_Band_mean = varargin{11};
                Bacon_Band_std = varargin{12};
                Bacon_BandRP_mean = varargin{13};
                Bacon_BandRP_std = varargin{14};
            end
        elseif strcmpi(x, 'no')
            disp('Unexpected inputs - Please enter all the arguments in proper order');
        else
            error('Please type "yes" or "no"');
        end
    otherwise
        error('Unexpected inputs - Please enter all the arguments');
end

% - Initializing the matrices to store randomized mean data.
bacon_rand_HRL = zeros(length(Bacon_HRL_mean),max(days));
bacon_rand_AGREL = zeros(length(Bacon_AGREL_mean),max(days));
bacon_rand_Q = zeros(length(Bacon_Q_mean),max(days));
bacon_rand_Band = zeros(length(Bacon_Band_mean),max(days));
bacon_rand_BandRP = zeros(length(Bacon_BandRP_mean),max(days));

% - Generating random numbers from a normal distribution formed using the
%   mean and the standard deviation of the collected results for various RL
%   Algorithms, as given inputs in lines .
for i = 1:length(Bacon_HRL_mean)
    bacon_rand_HRL(i,:) = normrnd(Bacon_HRL_mean(i),Bacon_HRL_std(i),[1 max(days)]);
end

for i = 1:length(Bacon_AGREL_mean)
    bacon_rand_AGREL(i,:) = normrnd(Bacon_AGREL_mean(i),Bacon_AGREL_std(i),[1 max(days)]);
end

for i = 1:length(Bacon_Q_mean)
    bacon_rand_Q(i,:) = normrnd(Bacon_Q_mean(i),Bacon_Q_std(i),[1 max(days)]);
end

for i = 1:length(Bacon_Band_mean)
    bacon_rand_Band(i,:) = normrnd(Bacon_Band_mean(i),Bacon_Band_std(i),[1 max(days)]);
end

for i = 1:length(Bacon_BandRP_mean)
    bacon_rand_BandRP(i,:) = normrnd(Bacon_BandRP_mean(i),Bacon_BandRP_std(i),[1 max(days)]);
end

% - Defining the publication specific plot settings.
hold on;
[~,~] = stdshade(bacon_rand_HRL',0.2,'m',days,0.01);
[~,~] = stdshade(bacon_rand_AGREL',0.2,'black',days,0.01);
[~,~] = stdshade(bacon_rand_Q',0.2,'c',days,0.01);
[~,~] = stdshade(bacon_rand_Band',0.2,'b',days,0.01);
[~,~] = stdshade(bacon_rand_BandRP',0.2,'g',days,0.01);

if nargin == 10
    plot(days,Bacon_Q_mean,'-.c*','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','c');
    plot(days,Bacon_HRL_mean,'-mv','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','m');
    plot(days,Bacon_AGREL_mean,'-k>','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','black');
    plot(days,Bacon_BandRP_mean,':gh','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','g');
    plot(days,Bacon_Band_mean,'-bd','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','b');
elseif nargin == 12
    plot(days,LDA_retrain,'-rs','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','red');
    plot(days,SVM_retrain,'-go','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','green');
    plot(days,LDA_fixed,'--b^','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','blue');
    plot(days,SVM_fixed,'-.gx','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','g');
    plot(days,Bacon_Q_mean,'-.c*','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','c');
    plot(days,Bacon_HRL_mean,'-mv','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','m');
    plot(days,Bacon_AGREL_mean,'-k>','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','black');
    plot(days,Bacon_BandRP_mean,':gh','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','g');
    plot(days,Bacon_Band_mean,'-bd','LineWidth',4,'MarkerSize',20,'MarkerFaceColor','b');
end

xticks(days);
yticks([0 20 40 60 80 100]);
xlabel('Recorded Sessions');
ylabel('Decoding Accuracy (%)');
title(expt_no);

ylim([0 100])
xlim([1 12])
ax = gca;
ax.FontSize = 40;
end


