% [fairMean, fairMeanAll,probCovAll,probAccAll]=...
%    funFairMeanDet(ppStructConfig,indexConfig,...
%    thresholdSINR,constNoise,muFading,funPathloss,funFair,S,theta,numbFeature)
%
% funFairMeanDet calculate the average fairness across an ensemble of
% network configurations.
%
% INPUTS:
%
% ppStructConfig is a structured array, where each element defines a network
% configuration or point pattern. For example, a single - element is created
% with the command:
% ppStructConfig = struct('xxTX',xxTX,'yyTX',yyTX,...
%        'xxRX',xxRX,'yyRX',yyRX,...
%        'n', numbPairs,'window',[xMin,xMax,yMin,yMax]);
% Here the arrays xxTX,yyTX,xxRX, and yyRX are the Cartesian coordinates
% of the transmitter-receiver n pairs, which are located in a square  with
% dimensions [xMin,xMax,yMin,yMax].
%
% indexConfig is index of the networks configurations being studied.
%
% thresholdSINR is the SINR threshold.
%
% constNoise is the noise constant.
%
% muFading is the mean of the iid random exponential fading variables.
%
% funPathloss is a single - variable function for the path - loss model
% eg funPathloss=@(r)(((1 + r)).^(-3.5));
%
% funFair is a single - variable function for the fairness model
% eg funFair=@(R)(log(R));
%
% OUTPUTS:
%
% fairMean is the fairness averaged over the network ensemble (ie all the
% network configurations).
%
% fairMeanAll is the total fairness for each network configuration.
%
% probCovAll is the coverage probability for each receiver.
%
% probAccAll is the access probability for each transmitter-receiver pair.
%
% This code was originally written by H.P. Keeler for the paper[1], which
% studies determinantal scheduling in wireless networks.
%
% If you use this code in published research, please cite paper[1].
%
% References:
%
% [1] B\laszczyszyn and Keeler, "Adaptive determinantal scheduling with
% fairness in wireless networks", 2025.
%
% [2] B\laszczyszyn, Brochard and Keeler, "Coverage probability in
% wireless networks with determinantal scheduling", 2020.
%
% Author: H. Paul Keeler, 2025.

function [fairMean, fairMeanAll,probCovAll,probAccAll]=...
    funFairMeanDet(ppStructConfig,indexConfig,...
    thresholdSINR,constNoise,muFading,funPathloss,funFair,S,theta,numbFeature)

numbConfig = numel(indexConfig); % number of network configurations
% number of pairs in each configuration
numbPairsAll = [ppStructConfig(indexConfig).n]';
numbPairsTotal = sum(numbPairsAll);% total number of pairs

% initialize  variable for average fairness in each training set
fairMeanAll = zeros(numbConfig,1);
% initialize  cell array for coverage probability
probCovAll = mat2cell(zeros(numbPairsTotal,1),numbPairsAll);
% initialize  cell array for access probability
probAccAll = mat2cell(zeros(numbPairsTotal,1),numbPairsAll);

% loop through for every training / learning sample
for tt = 1:numbConfig
    indexConfigTemp = indexConfig(tt);
    % retrieve x / y coordinates of all transmitter-receiver pairs
    xxTX = ppStructConfig(indexConfigTemp).xxTX;
    yyTX = ppStructConfig(indexConfigTemp).yyTX;
    xxRX = ppStructConfig(indexConfigTemp).xxRX;
    yyRX = ppStructConfig(indexConfigTemp).yyRX;

    % % create L matrix
    L = funPairsL(xxTX,yyTX,xxRX,yyRX,...
        S,theta,numbFeature);

    % calculate coverage probabilities for all transmitter-receiver pairs
    [probCovTemp,probAccTemp]=funProbCovPairsDetExact(xxTX,yyTX,xxRX,yyRX,...
        thresholdSINR,constNoise,muFading,funPathloss,L);
    probCovAll{tt}=probCovTemp; % coverage probability
    probAccAll{tt}=probAccTemp; % access probability

    rateTXRX = probCovTemp; % use coverage probability as rate

    % weight for "good" subsets; see funU definition
    weightConfig = 1;
    % calculate mean of fairness of rates
    fairMeanAll(tt)=mean(funFair(rateTXRX) * weightConfig);
end

fairMean = mean(fairMeanAll); % average fairness across all training sets

% check average fairness
if isinf(fairMean)
    warning(['The fairness is infinite, possibly due to exponentially '...
        'small coverage probabilities. Adjust network and SINR paramters.']);
end