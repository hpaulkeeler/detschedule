% [probCov,probTX,probCovCond]=funProbCovPairsDetExact(xxTX,yyTX,...,
%    xxRX,yyRX,thresholdSINR,constNoise,muFading,funPathloss,L,indexTransPair)
%
% Using closed-form expressions, the code calculates the coverage 
% probabilities in a bi-bole (or bipolar) network of transmitter-receiver 
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
% funPathloss is a single - variable function for the path loss model
% eg funPathloss=@(r)(((1 + r)).^(-3.5))
%
% L is the L - ensemble kernel, which can be created using the file
% funPairsL.m
%
% indexTransPair is an (optional) index for which the probabilities are 
% obtained. If the variable isn't passed, the code calculuates the probabilities
% for all the transmitter-receiver pairs.
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
% This code is a modification of that funProbCovPairsDet.m used
% originally for paper[2] by Blaszczyszyn, Brochard and Keeler.
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


function [probCov,probTX,probCovCond]=funProbCovPairsDetExact(xxTX,yyTX,...,
    xxRX,yyRX,thresholdSINR,constNoise,muFading,funPathloss,L,indexTransPair)

% reshape into column vectors
xxTX = xxTX(:);
yyTX = yyTX(:);
xxRX = xxRX(:);
yyRX = yyRX(:);

% helper functions; see the papers [1] and [2] for details.
% a helper function called the 'interference factor'.
fun_h=@(s,r)(1./((funPathloss(s)./funPathloss(r))...
    *thresholdSINR + 1));
% a helper function called the 'noise factor'.
fun_w=@(r)(exp(-(thresholdSINR / muFading) * constNoise./funPathloss(r)));

%%% START Numerical Connection Probability (ie SINR>thresholdConst) START%%%
K = funLtoK(L); % caclulate K kernel from kernel L
sizeK = size(K,1); % number of columns / rows in kernel matrix K

% calculate all respective distances (based on random network configuration)
% transmitters to other receivers
dist_ji_xx = bsxfun(@minus,xxTX,xxRX');
dist_ji_yy = bsxfun(@minus,yyTX,yyRX');
dist_ji = hypot(dist_ji_xx,dist_ji_yy); % Euclidean distances
% transmitters to receivers
dist_ii_xx = xxTX - xxRX;
dist_ii_yy = yyTX - yyRX;
dist_ii = hypot(dist_ii_xx,dist_ii_yy); % Euclidean distances
dist_ii = repmat(dist_ii',sizeK,1);% repeat cols for element - wise evaluation

% apply functions
hMatrix = fun_h(dist_ji,dist_ii); % matrix H for all h_{x_i}(x_j) values
W_x = fun_w(hypot(xxTX - xxRX,yyTX- yyRX)); % noise factor

if ~exist('indexTransPair','var')
    indexTransPair=(1:sizeK)'; % set index for all pairs
else
    indexTransPair = indexTransPair(:); % used supplied index
end

% transmitting - receiving probabilities are the diagonals of kernel
probTX = K(sub2ind([sizeK,sizeK],indexTransPair,indexTransPair));
probCovCond = zeros(length(indexTransPair),1);% coverage probability

% loop through for all pairs
for pp = 1:length(indexTransPair)
    indexTransTemp = indexTransPair(pp); % index of current pair

    % create h matrix corresponding to transmitter-receiver pair
    booleReduced = true(sizeK,1); % Boolean vector for all pairs
    booleReduced(indexTransTemp)=false;% remove transmitter
    % choose transmitter-receiver row
    hVectorReduced = hMatrix(booleReduced,indexTransTemp);
    % repeat vector hVectorReduced as columns
    hMatrixReduced = repmat(hVectorReduced,1,sizeK - 1);

    % create reduced Palm kernels
    KPalmReduced = funPalmK(K,indexTransTemp); % reduced Palm version of K matrix
    % calculate final kernel
    KReduced_h = sqrt(1 - hMatrixReduced').*KPalmReduced.*sqrt(1 - hMatrixReduced);

    % calculate unconditional probability for the event that transmitter's
    % signal at the receiver has an SINR>tau, given the pair is active (ie
    % transmitting and receiving)
    probCovCond(pp)=det(eye(sizeK - 1) - KReduced_h) * W_x(indexTransTemp);
end
% calculate unconditional probability
probCov = probCovCond.*probTX;
%%% END Numerical Connection Probability END%%%
end