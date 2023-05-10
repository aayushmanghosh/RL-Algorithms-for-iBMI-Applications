function [DecAcc,WiH,Wo] = HRL(testingSet,targetTest,rng_input)
% HRL Hebbian Reinforcement Learning Algorithm
%
%    HRL(testingSet,targetTest,rng_input) trains a multiclass
%    classifier according to the HRL ALgorithm. It also gives the
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

% Determining the weight matrices.
DecodingAcc_accum = [];

% Getting in the testing and training set
WiCell = {};
WoCell = {};
ind_W = 0;

for HLN = 75:25:200
    for mH =[0.0001 0.001 0.01 0.1 ]
        for mO = [0.0001 0.001 0.01 0.1]
            Decc_accumRng = [];
            ind_W = ind_W + 1;
            for rngChoice = rng_input
                % initialize the WiH matrix
                rng(rngChoice)
                WiH = randn(size(testingSet,2),HLN);
               
                % initialize the Wio matrix
                Wo = randn(HLN,max(targetTest));
                % evaluating the output
                errors = 0;
                for i=1:size(testingSet,1)
                    HiddenLayer = testingSet(i,:)*WiH;
                    HiddenLayer_tanh = tanh(HiddenLayer);
                    Output = tanh(sign(HiddenLayer_tanh)*Wo);
                    [~,I] = max(Output);
                    
                    if I == targetTest(i)
                        f = 1;
                    else
                        f = -1;
                        errors = errors + 1;
                    end
                    delWih = mH*f*testingSet(i,:)'*(sign(HiddenLayer_tanh)-HiddenLayer_tanh)...
                        + mH*(1-f)*(testingSet(i,:)'*(1 - sign(HiddenLayer_tanh) - HiddenLayer_tanh));
                    
                    WiH = WiH + delWih;
                    delWO = mO*f*HiddenLayer_tanh'*(sign(Output)-Output)...
                        + mO*(1-f)*(HiddenLayer_tanh'*(1 - sign(Output) - Output));
                    
                    Wo = Wo + delWO;
                end
                
                DecodingAcc = 1 - errors/size(testingSet,1);
                Decc_accumRng = [Decc_accumRng; DecodingAcc];
            end
            WiCell{ind_W,1} = WiH;
            WoCell{ind_W,1} = Wo;
            DecodingAcc_accum = [DecodingAcc_accum; mean(Decc_accumRng) std(Decc_accumRng) HLN mH mO];
        end
    end
end
[~,b] = max(DecodingAcc_accum(:,1));
DecAcc = DecodingAcc_accum(b,:);
WiH = WiCell{b,1};
Wo = WoCell{b,1};


