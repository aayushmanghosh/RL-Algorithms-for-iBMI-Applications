function model = banditron_multi_train(X,Y,model)
% BANDITRON_MULTI_TRAIN Banditron algorithm
%
%    MODEL = BANDITRON_MULTI_TRAIN(X,Y,MODEL) trains a multiclass
%    classifier according to the Banditron algorithm.
%
%    # Additional parameters:
%    - model.n_cla is the number of classes.
%    - model.gamma is the parameter that controls the trade-off between
%      exploration and exploitation.
%      Default value is 0.01.
%    
%    # Input:
%      - 'X': It is the training data matrix.
%      - 'Y': It is the matrix containing labels from 0 to whatever limit.
%      
%    # Output:
%      - 'model': The trained Banditron model.
%
%    # References:
%     - Kakade, S. M., Shalev-Shwartz, S., & Tewari, A. (2008).
%       Efficient bandit algorithms for online multiclass prediction.
%       Proceedings of the 25th International Conference on Machine
%       Learning (pp. 440-447).
%     - Ghosh A., and Shaikh S. et al., Lightweight Reinforcement
%       Learning Decoders for Autonomous, Scalable, Neuromorphic
%       intra-cortical Brain Machine Interface; submitted Neuromorphic
%       Computing & Interface, 2023.
%
%    # Version: v1.0
%    # Context: This file is part of the DOGMA library for MATLAB. This
%               function is used to generate the trained Banditron model. 
%               The output from the function is ultimately used to determine 
%               the decoding accuracy.
% License: Please see the accompanying file named "LICENSE"
% Author: Aayushman Ghosh, University of Illinois Urbana Champaign, May 2023.
%         <aghosh14@illinois.edu>

n = length(Y);   % number of training samples
% Defining the various model parameters that will be reflected in the
% output. The weight matrix, the totral error, the predicted classes. 
if isfield(model,'iter') == 0
    model.iter = 0;
    model.w = zeros(size(X,1),model.n_cla);
    model.errTot = zeros(numel(Y),1);
    model.numSV = zeros(numel(Y),1);
    model.aer = zeros(numel(Y),1);
    model.pred = zeros(model.n_cla,numel(Y));
end

% Defaulting value of the exploration constant is 0.01
if isfield(model,'gamma')==0
    model.gamma = .01;
end

for i=1:n
    model.iter = model.iter+1;
    Xi = X(:,i); % The training vector.
    val_f = model.w'*Xi; % Weight matrix multiplication.
    Yi = Y(i); % The accompanying label.
    
    [~,y_hat] = max(val_f); % Calculating the max
    % The following lines are listed based on the mathematics of the
    % Banditron Algorithm.
    Prob = zeros(1,model.n_cla)+model.gamma/model.n_cla; % The probability function.
    Prob(y_hat) = Prob(y_hat)+1-model.gamma; % Probability of he y_hat function.
    random_vect = (rand<cumsum(Prob));
    [~,y_tilde] = max(random_vect); % For exploration
    
    % The exploration parameters for the Banditron Algorithm. Refer to the
    % theory presented in the paper to better understand the underlying
    % algorithm of the Banditron.
    if model.iter>1
        model.errTot(model.iter) = model.errTot(model.iter-1)+(y_tilde~=Yi);
    else
        model.errTot(model.iter) + (y_tilde~=Yi);
    end
    model.aer(model.iter) = model.errTot(model.iter)/model.iter;
    model.pred(:,model.iter) = val_f;

    model.w(:,y_hat) = model.w(:,y_hat)-(Xi);
    
    if y_tilde==Yi
        model.w(:,Yi) = model.w(:,Yi)+(Xi)/Prob(y_tilde);   
    end
    
    model.numSV(model.iter) = numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f AER:%5.2f\n', ...
            ceil(i/1000),model.aer(model.iter)*100);
    end
end
