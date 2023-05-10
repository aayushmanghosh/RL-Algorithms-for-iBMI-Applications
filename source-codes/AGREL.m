function [DecAcc,WiH,Wo] = AGREL(testingSet,targetTest,rng_input)
% AGREL Attention Gated Reinforcement Learning Algorithm
%
%    AGREL(testingSet,targetTest,rng_input) trains a multiclass
%    classifier according to the AGREL ALgorithm. It also gives the
%    Decoding accuracy as the output along with the weights of the inner
%    layer to its connection to both the input and output layer.
%
%    # Inputs:
%      - 'testingSet': The input training matrix.
%      - 'targetTest': The true label set.
%      - 'rng_input': The random seed number (preferable value is 1).
%
%    # Outputs:
%      - 'DecAcc': The Decoding accuracy based on various Algorithms.
%      - 'WiH': The weights from input to hidden layer.
%      - 'Wo': The weights from hidden to output layer.
%
%    # References:%
%      - Ghosh A., and Shaikh S. et al., Lightweight Reinforcement
%      Learning Decoders for Autonomous, Scalable, Neuromorphic
%      intra-cortical Brain Machine Interface; submitted Neuromorphic
%      Computing & Interface, 2023.
%
%    # Version: v1.0
%    # Context: This function is used to generate the trained AGREL model.
%               The output from the function is ultimately used to determine
%               the decoding accuracy, and the weight matrices.
%
% License: Please see the accompanying file named "LICENSE"
% Author: Aayushman Ghosh, University of Illinois Urbana Champaign, May 2023.
%         <aghosh14@illinois.edu>

% Initializing an empty vector to calculate the Decoding accuracy.
DecodingAcc_accum = [];

% Determining the weight matrices.
WiCell = {};
WoCell = {};
ind_W = 0;

% Getting in the testing and training set
spike = testingSet';
n_timestep = length(targetTest);

for HLN = 75:25:200
    for learningRate = [0.0001 0.001 0.01 0.1]
        for exp_rate = [0.0001 0.001 0.01 0.1]
            Decc_accumRng = [];
            ind_W = ind_W + 1;
            for rngChoice = rng_input
                % initialize the WiH matrix
                rng(rngChoice)
                weightFromInputToHidden=(2*rand(HLN,size(testingSet,2))-1)/10;
                weightFromHiddenToOutput=(2*rand(max(targetTest),HLN)-1)/10;
                
                errors = 0;
                predictAction=[];
                rs = [];
                deltas = [];
                fdeltas = [];
                
                for i=1:n_timestep
                    % Calculating the output
                    inputUnit=spike(:,i);
                    hiddenUnit=mySigmoid(weightFromInputToHidden*inputUnit);
                    outputUnit=weightFromHiddenToOutput*hiddenUnit;
                    
                    % Select action based on softmax
                    pOutput = mySoftmax(outputUnit);
                    cumP = cumsum(pOutput);
                    rnd = rand();
                    % chooseAction = find(rnd<=cumP,1);
                    
                    % Choosing whether to explore, and choosing actions
                    % based on it.
                    explore = rand()<exp_rate;
                    if explore
                        chooseAction = randi([1,max(targetTest)],1);
                    else
                        [~,chooseAction] = max(pOutput);
                    end
                    if isempty(chooseAction)
                        chooseAction = 1;
                    end
                    predictAction= [predictAction;chooseAction];
                    
                    % use e-greedy policy
                    %                     pOutput = outputUnit;
                    %                     explore = rand()<exp_rate;
                    %                     if explore
                    %                         chooseAction = randi([1,max(targetTest_Acc)],1);
                    %                     else
                    %                         [~,chooseAction] = max(pOutput);
                    %                     end
                    %                     predictAction= [predictAction;chooseAction];
                    %
                    RewLiyuan = 0; RewShoeb = 1;
                    if RewLiyuan
                        currentPos = trajectory(i,:);
                        targetPos = target_location(i,:);
                        newPos = currentPos + vectors_scaled(chooseAction,:);
                        
                        currentDist = norm(targetPos - currentPos);
                        newDist = norm(targetPos - newPos);
                        changeDist = currentDist - newDist;
                        k = 5;
                        r = 1/(1+exp(-changeDist/k));
                        targetSize = params.targetSize;
                        targetReached = cursor_in_target(currentPos,targetPos,targetSize);
                        if  targetReached && chooseAction == 1
                            r = 1;
                        elseif targetReached && chooseAction ~= 1
                            r = 0;
                        elseif ~targetReached && chooseAction == 1
                            r = 0;
                        end
                    end
                    
                    if RewShoeb
                        if chooseAction == targetTest(i)% correct
                            r = 1;
                        else
                            r = 0;
                            errors = errors + 1;
                        end
                    end
                    if r == 1
                        delta = r - outputUnit(chooseAction);
                    else
                        delta = -1;
                    end
                    scale = 0; % fdelta scaling parameters.
                    if scale == 1
                        if delta > 0
                            fdelta = delta/(1-delta+1e-6);
                        else
                            fdelta = delta;
                        end
                    else
                        fdelta = delta;
                    end
                    
                    rs = [rs r];
                    deltas = [deltas delta];
                    fdeltas = [fdeltas fdelta];
                    
                    % Updating the weights
                    weightFromHiddenToOutput(chooseAction,:)=weightFromHiddenToOutput(chooseAction,:)+...
                        learningRate*fdelta*hiddenUnit';
                    weightFromInputToHidden=weightFromInputToHidden+...
                        learningRate*fdelta*...
                        (hiddenUnit.*(1-hiddenUnit).*weightFromHiddenToOutput(chooseAction,:)')*...
                        inputUnit';
                end
                DecodingAcc = 1 - errors/size(testingSet,1); % Collecting the Decoding Accuracies.
                Decc_accumRng = [Decc_accumRng; DecodingAcc];
            end
            
            WiCell{ind_W,1} = weightFromInputToHidden; % The Weight Matrices from input to hidden
            WoCell{ind_W,1} = weightFromHiddenToOutput; % The Weight Matrices from hidden to output
            DecodingAcc_accum = [DecodingAcc_accum; mean(Decc_accumRng) std(Decc_accumRng) HLN learningRate exp_rate];
        end
    end
end
[~,b] = max(DecodingAcc_accum(:,1));
DecAcc = DecodingAcc_accum(b,:);
WiH = WiCell{b,1};
Wo = WoCell{b,1};
end

