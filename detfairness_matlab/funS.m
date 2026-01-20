% S = funS(xx,yy,choiceKernel,paramKernel)
%
% This code generates a similarity matrix S based on Cartesian
% coordinates xx and yy.
%
% INPUTS:
%
% xx and yy are arrays for the Cartesian coordinates of the points.
%
% choiceKernel is a number between 1 and 3 for choosing which radial
% function for the similarity matrix. 1 is Gaussian, 2 is Cauchy, and 
% 3 is Bessel.
%
% paramKernel is an array of parameters for the radial functions. For the 
% aforementioned three radial functions, the first value is the scale 
% parameter sigma, while the second value is is a second paramter for the 
% Cauchy function. If the paramKernel = 0, then the identity matrix is returned 
% for the similarity matrix S.
%
% OUTPUTS:
%
% S is the similarity matrix.
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

function S = funS(xx,yy,choiceKernel,paramKernel)
% xx / yy need to be column vectors
xx = xx(:);
yy = yy(:);
sizeS = length(xx); % number of columns / rows

% retrieve kernel parameters
sigma = paramKernel(1);
alpha = paramKernel(end); % if there is a second parameter


distBetween = pdist([xx,yy]);% inter - point distances
distBetweenMean = mean(distBetween);% average inter - point distance
sigma = sigma * distBetweenMean; % rescale sigma

%%% NOTE:
% As sigma approaches zero, S approaches the identity matrix
% As sigma approaches infinity, S approaches a matrix of ones, which has a
% zero determinant (meaning its ill - conditioned in terms of inverses)

%%% START - Create similarity matrix S - START%%%
if sigma~=0
    % all squared distances of x / y difference pairs
    xxDiff = bsxfun(@minus,xx,xx'); 
    yyDiff = bsxfun(@minus,yy,yy');
    rrDiffSquared=(xxDiff.^2 + yyDiff.^2);

    if choiceKernel == 1
        %% Gaussian kernel
        % See the paper by Lavancier, Moller and Rubak (2015)
        S = exp(-(rrDiffSquared) / sigma^2);
    elseif choiceKernel == 2
        %% Cauchy kernel
        % See the paper by Lavancier, Moller and Rubak (2015)
        S = 1./(1 + rrDiffSquared / sigma^2).^(alpha + 1/2);
    elseif choiceKernel == 3
        %% Bessel kernel
        % See the Supplementary Material for the paper by  Biscio and
        % Lavancier (2016), page 2007. Kernel CI, where sigma has been
        % introduced as a scale parameter, similar to the Gaussian and
        % Cauchy cases.
        rrDiff = sqrt(rrDiffSquared);
        rrDiff(1:1 + size(rrDiff,1):end)=1; % prevent zero division
        % Bessel (simplified) kernel
        S = besselj(1,2 * sqrt(pi) * rrDiff / sigma)./(sqrt(pi) * rrDiff / sigma);
        % need to rescale to ensure that diagonal entries are ones.
        S(1:1 + size(S,1):end)=1; % set to correct value
    end
else
    % use identity matrix, which is the Aloha model
    S = eye(sizeS);
end
%%% END - Create similarity matrix S - END%%%
end