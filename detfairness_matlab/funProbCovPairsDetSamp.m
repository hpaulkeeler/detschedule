% [probCovEmp,probTXEmp,probCovCondEmp]=funProbCovPairsDetSamp(xxTX,yyTX,xxRX,yyRX,...
%    thresholdSINR,constNoise,muFading,funPathloss,L,numbSamp,indexTransPair)
%
% Using random simulations, the code estimates the coverage
% probabilities in a bi-pole (or bipolar) network of transmitter-receiver
% pairs based on the signal-to-interference-plus-noise ratio (SINR). The
% network has a random medium access control (MAC) scheme based on a
% determinantal point process, as outlined in the paper[1] by Blaszczyszyn
% and Keeler or the paper[2] by Blaszczyszyn, Brochard and Keeler.
%
% By coverage, it is assumed that each transmitter-receiver pair is active
% *and* the SINR of the transmitter is larger than some threshold at the
% corresponding receiver.
%
% If you use this code in published research, please cite paper[1].
%
% INPUTS:
%
% xxTX, yyTX are vectors for the Carestian coordinates of the transmitters.
% xxRX, yyRX are vectors for the Carestian coordinates of the receivers.
%
% thresholdSINR is the SINR threshold.
%
% constNoise is the noise constant.
%
% muFading is the mean of the iid random exponential fading variables.
%
% funPathloss is a single - variable function for the path - loss model
% eg funPathloss = @(r)(((1 + r)).^(-3.5));
%
% L is the L - ensemble kernel, which can be created using the file
% funPairsL.m
%
% indexTransmit is an (optional) index of the active (ie transmitting and
% receiving pairs). If it doesn't exist, the code assumes all pairs are
% active.
%
% OUTPUTS:
%
% probCov is the coverage probability of all the transmitter - receiver
% pairs.
%
% probTX is the medium access probability, meaning the probability that a
% pair is transmitting and receiving.
%
% probCovCond is the conditional probability that the transmitter
% has a SINR larger than some threshold at receiver.
%
% This code was used by H.P. Keeler for the paper[1] by
% Blaszczyszyn and Keeler, which studies determinantal scheduling in wireless
% networks.
%
%
% If you use this code in published research, please cite paper[1].
%
% References:
%
% [1] Blaszczyszyn and Keeler, "Adaptive determinantal scheduling with
% fairness in wireless networks", 2025.
%
% [2] Blaszczyszyn, Brochard and Keeler, "Coverage probability in
% wireless networks with determinantal scheduling", 2020.
%
% Author: H. Paul Keeler, 2025.

function [probCovEmp,probTXEmp,probCovCondEmp]=funProbCovPairsDetSamp(xxTX,yyTX,xxRX,yyRX,...
    thresholdSINR,constNoise,muFading,funPathloss,L,numbSamp,indexTransPair)

% Eigen decomposition of L
[eigenVecL,eigenValL]=eig(L);

% reshape into column vectors
xxTX = xxTX(:);
yyTX = yyTX(:);
xxRX = xxRX(:);
yyRX = yyRX(:);

sizeL = size(L,1); % number of columns / rows in kernel matrix L
numbPairs = sizeL;

if ~exist('indexTransPair','var')
    indexTransPair=(1:sizeL)'; % set index for all pairs
else
    indexTransPair = indexTransPair(:); % used supplied index
end

% initiate vectors for probability
probTXEmp = zeros(length(indexTransPair),1);
probCovCondEmp = zeros(length(indexTransPair),1);
probCovEmp = zeros(length(indexTransPair),1);

% loop through the network pairs
for pp = 1:length(indexTransPair)
    indexTransTemp = indexTransPair(pp); % index of current pair

    % transmitter location
    xxTX0 = xxTX(indexTransTemp);
    yyTX0 = yyTX(indexTransTemp);

    % receiver location
    xxRX0 = xxRX(indexTransTemp);
    yyRX0 = yyRX(indexTransTemp);

    %%% START Empirical Connection Probability (ie SINR>thresholdConst) START%%%
    % initialize  boolean vectors / arrays for collecting statistics
    booleTX = false(numbSamp,1); % transmitter-receiver pair exists
    booleCov = false(numbSamp,1); % transmitter-receiver pair is connected
    % loop through all simulations
    for ss = 1:numbSamp
        indexDPP = funSimSimpleDPP(eigenVecL,eigenValL);
        % if transmitter-receiver pair exists in the determinantal outcome
        booleTX(ss)=any(indexDPP == indexTransTemp);

        if booleTX(ss)
            % create Boolean variable for active interferers
            booleTransmit = false(numbPairs,1);
            booleTransmit(indexDPP)=true;
            booleTransmit(indexTransTemp)=false;

            % x / y values of interfering nodes
            xxInter = xxTX(booleTransmit);
            yyInter = yyTX(booleTransmit);

            % number of interferers
            numbInter = sum(booleTransmit);

            % simulate signal for interferers
            fadeRandInter = exprnd(muFading,numbInter,1); % fading
            distPathInter = hypot(xxInter - xxRX0,yyInter - yyRX0); % path distance
            proplossInter = fadeRandInter.*funPathloss(distPathInter); % path loss

            % simulate signal for transmitter
            fadeRandSig = exprnd(muFading); % fading
            distPathSig = hypot(xxTX0 - xxRX0,yyTX0 - yyRX0); % path distance
            proplossSig = fadeRandSig.*funPathloss(distPathSig); % path loss

            % Calculate SINR
            SINR = proplossSig / (sum(proplossInter) + constNoise);

            % see if transmitter is connected
            booleCov(ss)=SINR>thresholdSINR;
        end
    end
    % Estimate empirical probabilities
    probCovCondEmp(pp) = mean(booleCov(booleTX));% SINR>thresholdConst given pair
    probTXEmp(pp) = mean(booleTX); % transmitter-receiver pair exists
    probCovEmp(pp) = mean(booleCov); % SINR>thresholdConst
    %%% END Empirical Connection Probability (ie SINR>thresholdConst) END%%%
end
