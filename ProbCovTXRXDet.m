% Simulates a network with nodes, where each node can be either a
% transmitter or receiver (but not both) at any time step. The simulation
% examines the coverage based on the signal-to-interference ratio (SINR).
% The network has a random medium access control (MAC) scheme based on a
% determinantal point process, as outlined in the paper[1] by
% B\laszczyszyn, Brochard and Keeler. This code validates by simulation
% Propositions IV.1 and IV.2 in the paper[1]. This result gives the
% probability of coverage based on the SINR value of a transmitter-receiver
% pair in a non-random network of transmitter-or-receiver nodes such as a
% realization of a random point process.
%
% More specifically, the code estimates the probability of x and y being
% connected (ie SINR(x,y)>tau)given that x is transmitting and
% y isn't.
%
% The simulation section estimates the empirical probability of SINR-based
% coverage. For a large enough number of simulations, this empirical result
% will agree with the analytic results given in the paper[2].
%
% By coverage, it is assumed that the SINR of the transmitter is larger
% than some threshold at the corresponding receiver.
%
% Probabilities for other events are calculated/estimated including:
%
% Event A=SINR(x,y) > tau
% Event B=Transmitter exists
% Event C=Receiver exists
%
% This code was originally written by H.P Keeler for the paper by
% B\laszczyszyn, Brochard and Keeler[1].
%
% If you use this code in published research, please cite paper[1].
%
% References:
%
% [1] B\laszczyszyn, Brochard and Keeler, "Coverage probability in
% wireless networks with determinantal scheduling", 2020.
%
% Author: H. Paul Keeler, 2020.

close all;
clearvars; clc;

%set random seed for reproducibility
rng(1);

%%%START -- Parameters -- START%%%
choiceExample=1; %1 or 2 for a random (uniform) or deterministic example
numbSim=10^6; %number of simulations
numbNodes=10; %number of pairs
indexTrans=1; %index for transmitter
indexRec=2; %index for receiver
%above indices are bounded by numbNodes

%fading model
muFading=1/3; %Rayleigh fading average
%path loss model
betaPath=2; %pathloss exponent
kappaPath=1; %rescaling constant for pathloss function

thresholdSINR=0.1; %SINR threshold value
constNoise=0; %noise constant

%Determinantal kernel parameters
choiceKernel=1; %1 for Gaussian (ie squared exponetial );2 for Cauchy
%3 for independent (ie binomial) model
sigma=1;% parameter for Gaussian and Cauchy kernel
alpha=1;% parameter for Cauchy kernel
pAloha=0.5; %parameter for independent kernel (ie proportion transmitting)

%Simulation window parameters
xMin=-1; xMax=1; %x dimensions
yMin=-1; yMax=1; %y dimensions
xDelta=xMax-xMin; %rectangle width
yDelta=yMax-yMin; %rectangle height
xx0=mean([xMin,xMax]); %x centre of window
yy0=mean([yMin,yMax]); %y centre of window
%%%END -- Parameters -- END%%%

%Simulate a random point process for the network configuration
%interferer section
if choiceExample==1
    %random (uniform) x/y coordinates
    %transmitters or receivers
    xx=xDelta*(rand(numbNodes,1))+xMin;
    yy=yDelta*(rand(numbNodes,1))+yMin;
else
    %non-random x/y coordinates
    %transmitters or receivers
    t=2*pi*(linspace(0,(numbNodes-1)/numbNodes,numbNodes)');
    xx=(1+cos(5*t+1))/2;
    yy=(1+sin(3*t+2))/2;
end

%transmitter location
xxTX=xx(indexTrans);
yyTX=yy(indexTrans);

%Receiver location
xxRX=xx(indexRec);
yyRX=yy(indexRec);

%%% START -- CREATE L matrix -- START %%%
%all squared distances of x/y difference pairs
xxDiff=bsxfun(@minus,xx,xx');
yyDiff=bsxfun(@minus,yy,yy');
rrDiffSquared=(xxDiff.^2+yyDiff.^2);
if choiceKernel==1
    %%Gaussian/squared exponential kernel
    L=exp(-(rrDiffSquared)/sigma^2);
elseif choiceKernel==2
    %%Cauchy kernel
    L=1./(1+rrDiffSquared/sigma^2).^(alpha+1/2);
elseif choiceKernel==3
    %%Test kernel
    %%Independent model (equivalent to Aloha p)
    L=eye(numbNodes)*(pAloha/(1-pAloha));
else
    error('choiceKernel has to be equal to 1, 2, or 3.');
end
L=10*L; %scale matrix up (increases the eigenvalues ie number of points)
%%% END-- CREATE L matrix -- %%% END


%Eigen decomposition of L
[eigenVecL,eigenValL]=eig(L);

%Helper functions
funPathloss=@(r)((kappaPath*(1+r)).^(-betaPath)); %pathloss function
%The next functions are called interference factor and noise factor
%Note: the functions contain all the pathloss and SINR parameters
fun_h=@(s,r)(1./((funPathloss(s)./funPathloss(r))*thresholdSINR+1));
fun_w=@(r)(exp(-(thresholdSINR/muFading)*constNoise./funPathloss(r)));

%%%START Empirical Connection Proability (ie SINR>thresholdConst) START%%%
%initialize  boolean vectors/arrays for collecting statistics
booleA=false(numbSim,1); %transmitter is connected
booleB=false(numbSim,1); %transmitter exists
booleC=false(numbSim,1); %receiver exists

%loop through all simulations
for ss=1:numbSim
    %DPP for active transmitter nodes
    indexDPP=funSimSimpleDPP(eigenVecL,eigenValL);
    
    booleB(ss)=any(indexDPP==indexTrans); %if transmitter is in subset
    booleC(ss)=all(indexDPP~=indexRec); %if receiver is not in subset
    
    %if transmitter is in the determinantal subset, calculate its SINR
    if booleB(ss)
        %create Boolean variable for active interferers
        booleInter=false(numbNodes,1);
        booleInter(indexDPP)=true;
        booleInter(indexTrans)=false; %exclude transmitter
        
        %x/y values of interfering nodes
        xxInter=xx(booleInter);
        yyInter=yy(booleInter);
        
        %number of interferers
        numbInter=sum(booleInter);
        
        %simulate signal for interferers
        fadeRandInter=exprnd(muFading,numbInter,1); %fading
        distPathInter=hypot(xxInter-xxRX,yyInter-yyRX); %path distance
        proplossInter=fadeRandInter.*funPathloss(distPathInter); %pathloss
        
        %simulate signal for transmitter
        fadeRandSig=exprnd(muFading); %fading
        distPathSig=hypot(xxTX-xxRX,yyTX-yyRX); %path distance
        proplossSig=fadeRandSig.*funPathloss(distPathSig); %pathloss
        
        %Calculate the SINR
        SINR=proplossSig/(sum(proplossInter)+constNoise);
        
        %see if transmitter is connected
        booleA(ss)=SINR>thresholdSINR;
    end
end
booleBandC=booleB&booleC; %transmitter-receiver pair exists
booleNotC=~booleC; %receiver does not exist
booleBandNotC=booleB&booleNotC; %transmitter exists, receiver does not

%%%START Create kernels and Palm kernels START%%%
K=funLtoK(L); %caclulate K kernel from kernel L
sizeK=size(K,1); %number of columns/rows in kernel matrix K

%Calculate all respective distances (based on random network configuration)
%from all transmitters to receiver
dist_ji_xx=bsxfun(@minus,xx,xxRX');
dist_ji_yy=bsxfun(@minus,yy,yyRX');
dist_ji=hypot(dist_ji_xx,dist_ji_yy); %Euclidean distances
%transmitters to receivers
dist_ii_xx=xxTX-xxRX;
dist_ii_yy=yyTX-yyRX;
dist_ii=hypot(dist_ii_xx,dist_ii_yy); %Euclidean distances
dist_ii=repmat(dist_ii',sizeK,1);%repeat cols for element-wise evaluation

%apply functions
hMatrix=fun_h(dist_ji,dist_ii); %matrix H for all h_{x_i}(x_j) values
W_x=fun_w(hypot(xx-xxRX,yy-yyRX)); %noise factor

%create h matrix corresponding to transmitter
booleAll=true(sizeK,1); %Boolean vector for all nodes
booleReduced=booleAll;
booleReduced(indexTrans)=false;%remove transmitter

%choose transmitter row
hVectorReduced=hMatrix(booleReduced,indexTrans);
%repeat vector hVectorReduced as columns
hMatrixReduced=repmat(hVectorReduced,1,sizeK-1);

%create Palm kernels conditioned on transmitter existing
[KPalmReducedTX,KPalmTX] =funPalmK(K,indexTrans);
%create Palm kernels conditioned on receiver existing
[KPalmRXReduced,KPalmRX]=funPalmK(K,indexRec);
%create Palm kernels conditioned on  transmitter AND receiver existing
[~,KPalmTXRX]=funPalmK(KPalmTX,indexRec);
%create reduced (by transmitter) Palm kernel conditioned on transmitter
%AND receiver existing
KPalmSemiReducedTXRX=KPalmTXRX(booleReduced,booleReduced);

%calculate final kernels
%for transmitter
KReduced_hTX=sqrt(1-hMatrixReduced').*KPalmReducedTX...
    .*sqrt(1-hMatrixReduced);
%for reciever and transmitter
KReduced_hRX=sqrt(1-hMatrixReduced').*KPalmSemiReducedTXRX...
    .*sqrt(1-hMatrixReduced);
%%%END Create kernels and Palm kernels END%%%

%%%START Connection Proability (ie SINR>thresholdConst) START%%%
%calculate probabiliity for the event that transmitter's
%signal at the receiver has an SINR>thresholdConst, given the pair is 
% active (ie trasnmitting and receiving); see Section IV in paper[1].

%probability transmitter exists (ie transmitter at indexTrans) - event B
probB=K(indexTrans,indexTrans)
probB_Emp=mean(booleB)

%probability receiver exists (ie no transmitter at indexRec) - event C
probC=1-K(indexRec,indexRec)
probC_Emp=mean(booleC)

%probability transmitter but no receiver
probBNotC=det(K([indexTrans,indexRec],[indexTrans,indexRec]))
probBNotC_Emp=mean(booleBandNotC)

%probability transmitter and receiver existing
probBandC=probB-probBNotC
probBandC_Emp=mean(booleBandC)

%probability of SINR>threshold (ie transmiter is connected ) given B
probA_GivenB=det(eye(sizeK-1)-KReduced_hTX)*W_x(indexTrans)
probA_GivenB_Emp=mean(booleA(booleB))

%probability of SINR>threshold (ie transmiter is connected ) given B and C
probA_GivenBNotC=det(eye(sizeK-1)-KReduced_hRX)*W_x(indexTrans)
probA_GivenBNotC_Emp=mean(booleA(booleNotC))

%probability B given NOT C (ie a transmitter exists at indexRec)
probB_GivenNotC=KPalmRX(indexTrans,indexTrans)
probB_GivenNotC_Emp=mean(booleB(booleNotC))

%probability B given C
probB_GivenC=(probB-(1-probC)*probB_GivenNotC)/probC
probB_GivenC_Emp=mean(booleB(booleC))

%probability NOT C (ie a transmitter exists at indexRec) given B
probNotC_GivenB=KPalmTX(indexRec,indexRec)
probNotC_GivenB_Emp=mean(booleNotC(booleB))

%probability C given B
probC_GivenB_Emp=mean(booleC(booleB))
probC_GivenB=1-probNotC_GivenB

disp('Conditional coverage probability (ie A given B and C).')
%coverage probability ie probability of A given B and C
probA_GivenBandC=(probA_GivenB-probNotC_GivenB*probA_GivenBNotC)...
    /probC_GivenB

%Estimate empirical probability two different ways
%Directly
probA_GivenBandC_Emp1=mean(booleA(booleBandC))
%Indirectly
probA_GivenBandC_Emp2=(probA_GivenB_Emp-probNotC_GivenB_Emp*probA_GivenBNotC_Emp)...
    /probC_GivenB_Emp;

disp('Coverage probability (ie A given B and C).')
%connection probability
probCov=probA_GivenBandC*probBandC
probCov_Emp1=mean(booleA&booleB&booleC)
%probCov_Emp2=probA_GivenBandC_Emp2*probBandC_Emp

%probCovCond=probA_GivenBandC %conditional coverage probability
%probTXRX=probBandC %probability of pair existing
%connection probability
%probCov=probCovCond*probTXRX

%%%END Connection Proability (ie SINR>thresholdConst) END%%%

%TEST
[probCov,probTXRX,probCovCond]=funProbCovTXRXDet(xx,yy,...
   fun_h,fun_w,L,indexTrans,indexRec)


if ~isempty(indexDPP)
    %%% START -- Plotting -- START %%%
    figure;hold on;
    markerSizeNumb=80; %marker size of markers colors
    vectorColor=rand(1,3).^(1); %random vector for colors of
    %Plot point process
    plot(xx,yy,'ko','MarkerSize',markerSizeNumb/6);
    %Plot determinantally-thinned point process
    plot(xx(indexDPP),yy(indexDPP),'.','MarkerSize',markerSizeNumb/3,'color',vectorColor);
    grid;
    axis square;set(gca,'YTick',[]); set(gca,'XTick',[]);
    legend('Original point process', 'Determinantal subset');
    %%% END -- Plotting -- END %%%
end