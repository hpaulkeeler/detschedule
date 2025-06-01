% function [L,q]=funPairsL(xxTX,yyTX,xxRX,yyRX,S,theta,numbFeature,...
%    fun_q)
%
% This function creates an L-ensemble kernel matrix L and a quality vector
% q as detailed in the paper[1] by Blaszczyszyn, Brochard and Keeler.
%
% INPUTS:
%
% xxTX, yyTX are vectors for the Carestian coordinates of the transmitters.
% xxRX, yyRX are vectors for the Carestian coordinates of the receivers.
%
% S is the similarity matrix, which creates replusion among the
% points, can be formed from either, Gaussian, Cauchy or Bessel kernel
% function. See funPairsS.m for more details.
%
% theta is is array representing a fitting parameter for the quality model
% q(theta,f), where f is a vector of features with the same dimensions as theta, 
% In this code, features are the distsances of the first and second nearest 
% neighbouring point. The quality model is convex, for example:
%
% q(theta,f) = |theta*f'|^2
%
% numbFeature is a single number (0,1, or 2) representing the number of
% features to be used in the quality model q. If numbFeature = 0, then no
% features are used in the quality model, and the quality model becomes
% q=fun_q(theta).
%
% fun_q(tf) is a single-variable (optional) function for the quality model,
% where tf = theta.*f or tf = theta.
%
% OUPUTS:
%
% L is an L-ensemble kernel matrix L.
%
% q is an array representing the vector quality model
%
% This code was originally written by H.P. Keeler for the paper[1] by
% Blaszczyszyn and Keeler, which studies determinantal scheduling in wireless
% networks.
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
%
%%%%% NEEDS FURTHER CLEANING UP AND COMMMENTING %%%%%%



function [L,q]=funPairsL(xxTX,yyTX,xxRX,yyRX,S,theta,numbFeature,...
    fun_q)

numb_theta=numel(theta); %number of elements in theta

if exist('fun_q','builtin')
    fun_quality=@(tf)fun_q(tf); %use passed quality model
else
    %create default quality model
    %use q(theta,f)=|theta.*f|^p model, where .* is a dot/scalar product.
    %NOTE p needs to be p>=2 to ensure a convex function
    p=2;
    fun_quality=@(tF)(abs(tF).^p);

    %NOTE: Possible to use q=exp(theta), but it seems to be numerically
    %unstable, as it creates large numbers and is hard to fit with optimization
    %functions such as fminunc
end

if numbFeature==0
    numbFeature=numb_theta;
end
if numb_theta==0
    error('theta needs at least one element.');
end
if numbFeature>numb_theta
    error('Not enough elements in theta.')
end
if (numbFeature>1)&&(numbFeature>numel(xxTX))
    error('Need more points for theta vector of given length.')
end

%Convert any matrices into vectors and rescale
theta=theta(:); %theta needs to be columm vector

%x/y coordinates need to be column vectors
xxTX=xxTX(:);
yyTX=yyTX(:);
xxRX=xxRX(:);
yyRX=yyRX(:);

sizeL=length(xxTX); %width/height of L (ie cardinality of state space)

%%%START - Create q (ie quality feature/covariate) vector START%%%
% variable thetaFeature stores theta*f, where theta is the fitting
% (sclaar or vector) parameter and f is a feature vector.

if numbFeature==0
    %no features means just the theta parameter
    thetaFeature=theta;
else
    %zeroth term
    feature_1=ones(sizeL,1);
    thetaFeature=theta(1)*feature_1;
    %non-zeroth terms
    if numbFeature>1
        %for each transmitter, calculate distances to all other receivers
        dist_ji_xx=bsxfun(@minus,xxTX,xxRX');
        dist_ji_yy=bsxfun(@minus,yyTX,yyRX');
        dist_ji=hypot(dist_ji_xx,dist_ji_yy); %Euclidean distances

        dist_jj=diag(dist_ji); %Euclidean distances between pairs
        dist_jj=repmat(dist_jj',sizeL,1);%repeat cols for element-wise evaluation
        dist_ji=dist_ji./dist_jj; %rescale by transmitter-receiver distance
        dist_ji(logical(eye(sizeL)))=Inf;  %set diagonals to infinity

        if numbFeature==2
            feature_2=min(dist_ji,[],2); %find minimums across each row
            theta_feature_2=theta(2)*feature_2;
            %add contribution from parameters and features
            thetaFeature=thetaFeature+theta_feature_2;
        else
            %sort distances in ascending order
            dist_ji_sorted=sort(dist_ji,2);
            feature_n=dist_ji_sorted(:,1:(numbFeature-1));
            %replicate parameter vector
            thetaVector_n=repmat(theta(2:end),1,sizeL);
            %element-wise product of parameters and features
            thetaFeature_n=(thetaVector_n').*feature_n;
            %sum for each transmitter (ie giving dot product)
            thetaFeature=thetaFeature+sum(thetaFeature_n,2);
        end

    end
end

%apply quality model; see fun quality
q=fun_quality(thetaFeature);
%%%END - Create q (ie quality feature/covariage) vector END%%%

%START Create L matrix
qMatrix=repmat(q',size(S,1),1); %q diagonal matrix
L=(qMatrix').*S.*qMatrix;
%END Create L matrix

end
