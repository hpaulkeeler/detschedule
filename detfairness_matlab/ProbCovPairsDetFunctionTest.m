% Simulates a network with transmitter-receiver pairs in order to examine
% the coverage based on the signal-to-interference-plus-ratio ratio (SINR).  
% The bi-pole network has a random medium access control (MAC) scheme based on a
% determinantal point process, as outlined in the paper[1] by
% Blaszczyszyn, Brochard and Keeler. This code validates by simulation
% Propositions III.1 and III.2 in the paper[1]. These results give the 
% probability of coverage based on the SINR value of a transmitter-receiver
% pair in a non - random network of transmitter-receiver pairs such as a 
% realization of a random point process.
%
% The simulation section estimates the empirical probability of SINR-based
% coverage. For a large enough number of simulations, this empirical result
% will agree with the analytic result given by Propositions III.1 and III.2
% in the paper[1].
%
% By coverage, it is assumed that the transmitter-receiver pair are active
% *and* the SINR of the transmitter is larger than some threshold at the
% corresponding receiver.
%
% It was originally written by H.P. Keeler for the paper[2] by Blaszczyszyn, 
% Brochard and Keeler.
%
% If you use this code in published research, please cite the paper[1] by
% Blaszczyszyn and Keeler.
%
% More details are given in paper[1] and paper[2]. Also see the book[3] by Taskar
% and Kulesza.
%
% References:
%
% [1] Blaszczyszyn and Keeler, "Adaptive determinantal scheduling with
% fairness in wireless networks", 2025.
%
% [2] Blaszczyszyn, Brochard and Keeler, "Coverage probability in
% wireless networks with determinantal scheduling", 2020.
%
% [3] Taskar and Kulesza, "Determinantal point processes for machine learning", 2012.
%
% Author: H. Paul Keeler, 2025.

%close all;
%clearvars; clc;

% set random seed for reproducibility
rng(1);

%%% START -- Parameters -- START%%%
% configuration choice
choiceExample = 2; % 1 or 2 for a random (uniform) or deterministic example
numbSim = 10^4; % number of simulations
numbPairs = 5;% number of pairs
indexTransPair = 1; % index for transmitter-receiver pair
% the above index has to be such that 1<=indexTransPair<=numbPairs

% fading model
muFading = 1 / 3; % Rayleigh fading average
% path loss model
betaPath = 2; % pathloss exponent
kappaPath = 1; % rescaling constant for pathloss function
% SINR model
thresholdSINR = 0.1; % SINR threshold value
constNoise = 0; % noise constant

% choose kernel
choiceKernel = 1; % 1 for Gaussian (ie squared exponetial );2 for Cauchy
sigma = 0;% parameter for Gaussian and Cauchy kernel
alpha = 1;% parameter for Cauchy kernel

% simulation window parameters
% x / y dimensions
xMin = -1;
xMax = 1; 
yMin = -1;
yMax = 1; 
% width / height
xDelta = xMax - xMin; 
yDelta = yMax - yMin;
% x / y centre of window
%%% END -- Parameters -- END%%%

% Simulate a random point process for the network configuration
% interferer section
if choiceExample == 1
    % random (uniform) x / y coordinates
    % transmitters
    xxTX = xDelta * (rand(numbPairs,1)) + xMin;
    yyTX = yDelta * (rand(numbPairs,1)) + yMin;
    % all receivers
    xxRX = 2 * xDelta * (rand(numbPairs,1)) + xMin;
    yyRX = 2 * yDelta * (rand(numbPairs,1)) + yMin;
else
    % non - random x / y coordinates
    % all transmitters
    t = 2 * pi * (linspace(0,(numbPairs - 1) / numbPairs,numbPairs)');
    xxTX=(1 + cos(5 * t+1)) / 2;
    yyTX=(1 + sin(3 * t+2)) / 2;
    % all receivers
    xxRX=(1 + sin(3 * t+1)) / 2;
    yyRX=(1 + cos(7 * t+2)) / 2;
end

% transmitter location
xxTX0 = xxTX(indexTransPair);
yyTX0 = yyTX(indexTransPair);

% receiver location
xxRX0 = xxRX(indexTransPair);
yyRX0 = yyRX(indexTransPair);

%%% START -- CREATE L matrix -- START %%%
% all squared distances of x / y difference pairs
xx = xxTX;
yy = yyTX;
xxDiff = bsxfun(@minus,xx,xx');
yyDiff = bsxfun(@minus,yy,yy');
rrDiffSquared=(xxDiff.^2 + yyDiff.^2);

if sigma~=0
if choiceKernel == 1
    % Gaussian kernel
    L = exp(-(rrDiffSquared) / sigma^2);
elseif choiceKernel == 2
    % Cauchy kernel
    L = 1./(1 + rrDiffSquared / sigma^2).^(alpha + 1/2);
else
    error('choiceKernel has to be equal to 1 or 2.');
end
else
    % use identity matrix, which is the Aloha model
    L = eye(numbPairs);
end

L = 10 * L; % scale matrix up (increases the eigenvalues ie number of points)
%%% END-- CREATE L matrix -- %%% END

% eigen decomposition of L
[eigenVecL,eigenValL]=eig(L);

% helper functions
funPathloss=@(r)((kappaPath * (1 + r)).^(-betaPath)); % pathloss function

[probCov,probTX,probCovCond] = funProbCovPairsDetExact(xxTX,yyTX,...
    xxRX,yyRX,thresholdSINR,constNoise,muFading,funPathloss,L,indexTransPair)

[probCovEmp,probTXEmp,probCovCondEmp] = funProbCovPairsDetSamp(xxTX,yyTX,xxRX,yyRX,...
thresholdSINR,constNoise,muFading,funPathloss,L,numbSim,indexTransPair)