% [L,q]=funPairsL(xxTX,yyTX,xxRX,yyRX,S,theta,numbFeature, fun_q)
%
% This function creates an L - ensemble kernel matrix L and a quality vector
% q as detailed in the paper[1] by Blaszczyszyn and Keeler.
%
% INPUTS:
%
% xxTX, yyTX are vectors for the Cartesian coordinates of the transmitters.
% xxRX, yyRX are vectors for the Cartesian coordinates of the receivers.
%
% S is the similarity matrix, which creates repulsion among the
% points, can be formed from either, Gaussian, Cauchy or Bessel kernel
% function. See the function funS in the file funS.m for more details.
%
% theta is a vector (or array) representing a fitting parameter for the quality 
% model q(theta,f), where f is a vector of features with the same dimensions as 
% theta. Here the features are the distances of k - th nearest neighbouring points. 
% The quality model is typically a convex function of the matrix product of theta 
% and the transpose of f, which we write here as f'. For example:
%
% q(theta,f) = |theta * f'|^2
%
% numbFeature is a single non - negative integer representing the number of
% features to be used in the quality model q. If numbFeature = 0, then no
% features are used in the quality model, and the quality model becomes
% q(theta,f) = q(theta).
%
% fun_q(t) is a single - variable (optional) function for the quality model,
% where t = theta * f' or t = theta. If no function is given, then the default 
% function fun_q(t) = |t|^2 is used.
%
% OUTPUTS:
%
% L is an L - ensemble kernel matrix L.
%
% q is an array representing the vector quality model.
%
% This code was originally written by H.P. Keeler for the paper[1] by
% Blaszczyszyn and Keeler, which studies determinantal scheduling in wireless
% networks.
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

function [L,q]=funPairsL(xxTX,yyTX,xxRX,yyRX,S,theta,numbFeature,...
    fun_q)
numb_theta = numel(theta); % number of elements in theta

if ~exist('fun_q','builtin')
    % create default quality model, which is of the form:
    % (theta,f)=|theta * f'|^p model.
    % NOTE p needs to be p>=2 to ensure a convex function
    fun_q=@(t)(abs(t).^2);
    % NOTE: Possible to use q = exp(theta), but it seems to be numerically
    % unstable, as it creates large numbers and is hard to fit with standard 
    % optimization functions.
end

if numbFeature == 0
    numbFeature = numb_theta;
end
if numb_theta == 0
    error('theta needs at least one element.');
end
if numbFeature>numb_theta
    error('Not enough elements in theta.')
end
if (numbFeature>1)&&(numbFeature>numel(xxTX))
    error('Need more points for theta vector of given length.')
end

% convert any matrices into vectors and rescale
theta = theta(:); % theta needs to be columm vector

% x / y coordinates need to be column vectors
xxTX = xxTX(:);
yyTX = yyTX(:);
xxRX = xxRX(:);
yyRX = yyRX(:);

sizeL = length(xxTX); % width / height of L (ie cardinality of state space)

%%% START - Create q (ie quality feature / covariate) vector START %%%
% variable thetaFeature stores theta * f, where theta is the fitting
% (scalar or vector) parameter and f is a feature vector.

if numbFeature == 0
    % no features means just the theta parameter
    thetaFeature = theta;
else
    % zeroth term
    feature_1 = ones(sizeL,1);
    thetaFeature = theta(1) * feature_1;
    % non - zeroth terms
    if numbFeature>1
        % for each point, calculate distances to all other points
        dist_ji_xx = bsxfun(@minus,xxTX,xxRX');
        dist_ji_yy = bsxfun(@minus,yyTX,yyRX');
        dist_ji = hypot(dist_ji_xx,dist_ji_yy); % Euclidean distances
        dist_jj = diag(dist_ji); % Euclidean distances between pairs
        dist_jj = repmat(dist_jj',sizeL,1);% repeat cols for element - wise evaluation
        dist_ji = dist_ji./dist_jj; % rescale by point - to - point distance
        dist_ji(logical(eye(sizeL)))=Inf;  % set diagonals to infinity

        if numbFeature == 2
            feature_2 = min(dist_ji,[],2); % find minimums across each row
            theta_feature_2 = theta(2) * feature_2;
            % add contribution from parameters and features
            thetaFeature = thetaFeature + theta_feature_2;
        else
            % sort distances in ascending order
            dist_ji_sorted = sort(dist_ji,2);
            feature_n = dist_ji_sorted(:,1:(numbFeature - 1));
            % replicate parameter vector
            thetaVector_n = repmat(theta(2:end),1,sizeL);
            % element - wise product of parameters and features
            thetaFeature_n=(thetaVector_n').*feature_n;
            % sum for each point (ie giving dot product)
            thetaFeature = thetaFeature + sum(thetaFeature_n,2);
        end

    end
end

% apply quality model
q = fun_q(thetaFeature);
%%% END - Create q (ie quality feature / covariate) vector END%%%

% START Create L matrix
qMatrix = repmat(q',size(S,1),1); % q diagonal matrix
L=(qMatrix').*S.*qMatrix;
% END Create L matrix

end
