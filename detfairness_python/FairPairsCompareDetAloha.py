# The code here was used to generate the numerical results in the
# paper[1] by Blaszczyszyn and Keeler.
#
# The code's purpose is studying determinantal scheduling (described below)
# when maximizing the coverage probability, which is defined as the tail
# distribution of the signal-to-interference-plus-noise ratio (SINR), or
# a function of the coverage probability, which is called fairness.
#
# This code generates network configurations of n transmitter-receiver
# pairs (x_1,y_1),...,(x_n,y_n), which is known as the bi-pole or
# bi - polar network model. The transmitters x_1,...,x_n are scattered
# uniformly. Each receiver y_i is located at random or fixed distance
# from transmitter x_i, whereas the angle of orientation is a uniform
# random variable.
#
# Using the coverage probability, defined as P(SINR(x_i,y_i)>tau) where
# tau>0, as the rate, the code then finds the optimal access/transmitting
# probability for three separate random scheduling algorithms:
#
# Fixed Aloha: Each transmitter-receiver pair (x_i,y_i) is independently
# active (so the transmitter is transmitting) with fixed probability p.
# The active pairs form a binomial point process.
#
# Adaptive Aloha: Each transmitter-receiver pair (x_i,y_i) is
# independently active with probability p_i, where p_i values can vary.
# This scheduling scheme generalizes the fixed Aloha scheme.
#
# Determinantal: Each transmitter-receiver pair (x_i,y_i) is active with
# probability K_ii, where K_ii is the i - th diagonal of the determinantal
# kernel matrix K, which is related to the L matrix by K = L / (I + L), where I
# is an identity matrix. The active pairs form a determinantal point
# process. This scheduling scheme generalizes the adaptive Aloha scheme.
#
# When finding the optimal acccess probabilities, the code either
# maximizes the total throughput (defined as the SINR coverage
# probability of a receiver) or the total fairness (defined as the log of
# the throughput).
#
# The logarithmic fairness function results in proportional fairness, as
# studied in the paper[4] by Kelly, Maulloo and Tan, but other fairness
# functions can be used.
#
# This code was originally written by H.P. Keeler for the paper[1] by
# Blaszczyszyn and Keeler, which studies determinantal scheduling with
# proportional fairness in wireless networks.
#
# If you use this code in published research, please cite the paper[1] by
# Blaszczyszyn and Keeler.
#
# More details are given in paper[1] and paper[2]. Also see the book[3] by Taskar
# and Kulesza.
#
# References:
#
# [1] Blaszczyszyn and Keeler, "Adaptive determinantal scheduling with
# fairness in wireless networks", 2025.
#
# [2] Blaszczyszyn, Brochard and Keeler, "Coverage probability in
# wireless networks with determinantal scheduling", 2020.
#
# [3] Taskar and Kulesza, "Determinantal point processes for machine learning", 2012.
#
# [4] Kelly, Maulloo, and Tan, "Rate control for communication networks:
# shadow prices, proportional fairness and stability", 1998.
#
# Author: H. Paul Keeler, 2025.

import numpy as np; # NumPy package for arrays, random number generation, etc
import matplotlib.pyplot as plt # for plotting
from matplotlib import collections  as mc # for plotting line segments
from scipy.optimize import minimize # for optimizing
import warnings; # for giving warning message
# import time; # for timing functions etc

from funPairsL import funPairsL # generate L matrix
from funProbCovPairsDetExact import funProbCovPairsDetExact  # coverage probability
from funFairMeanDet import funFairMeanDet
from funS import funS; # similarity matrix

# set display / print options to resemble those of MATLAB
np.set_printoptions(precision = 4,suppress = True);

plt.close("all");  # close all figures

# set random seed for reproducibility
np.random.seed(1);

### START -- Parameters -- START ###
numbSim = 10; # number of simulations
numbPairs = 5; # number of transmitter-receiver pairs
booleFair = 1; # 0 for no fairness, 1 for fairness with utility function
numbPlot = 1; # 1 just plot coverage probabilities, 2 plot also fairness values
boolePlotAverage = 1; # plot ensemble averages

# 1 for a random global network model;
# 2 / 3 for local with fixed / random transmitter-receiver distance
choiceNetwork = 1; # choose a network model
rMax = .2; # max transmitter-receiver distance

# fading model
muFading = 1; # Rayleigh fading average
# path loss model
betaPath = 4; # path loss exponent
kappaPath = 1; # rescaling constant for path loss function

# SINR model
thresholdSINR = 1; # SINR threshold value
constNoise = 0; # noise constant

# choose kernel for determinantal point process
choiceKernel = 1; # 1 for Gaussian;2 for Cauchy
sigma = 10; # parameter for Gaussian and Cauchy kernel
alpha = 1; # parameter for Cauchy kernel

paramKernel = np.array([sigma,alpha]);

numbFeature = 0; # ranging from 0 to n (if numbFeature>0 requires n - 1 points)
# set numbFeature = 0 to optimize the quality feature q_i for each
# transmitter location x_i

optionsOpt={'disp': False}; # options for minimize

# Simulation window parameters
# x / y dimensions
xMin = -.5;
xMax = .5;
yMin = -.5;
yMax = .5;
# rectangle width / height
xDelta = xMax - xMin;
yDelta = yMax - yMin;
# x / y centre of window
xx0 = np.mean(np.array([xMin,xMax]));
yy0 = np.mean(np.array([yMin,yMax]));

numbModelComp = 3; # number of models to compare
# determinantal, adaptive Aloha, fixed Aloha
### END -- Parameters -- END ###

# helper functions
def funPathloss(r):
    return (kappaPath * (1 + r))**(-betaPath);  # path loss function
# for checking medium access / transmitting probability pAccess
fun_q_to_p = lambda q:(q * q / (1 + q**2));
fun_p_to_q = lambda p:(np.sqrt(p / (1 - p)));
def fun_pol2cart(theta,rho):
    return rho * np.cos(theta),rho * np.sin(theta) #polar to Cartesian coordinates

if booleFair:
    # fairness function
    def funFair(R):
        return (np.log(R)); # fairness function eg log(R) or 1 / R^alpha
    labelTitle="Maximizing throughput utility U(T) = log(T)";
else:
    def funFair(R):
        return (R);
    labelTitle="Maximizing throughput T";

labelTitle = labelTitle + ". SINR threshold = " + str(thresholdSINR);

# initiate arrays for collecting statistics
# coverage probabilities
probCovAllDet = np.zeros((numbSim,numbPairs));
probCovAllAlohaA = np.zeros((numbSim,numbPairs));
probCovAllAlohaF = np.zeros((numbSim,numbPairs));
# transmitting probabilities
probTX_AllDet = np.zeros((numbSim,numbPairs));
probTX_AllAlohaA = np.zeros((numbSim,numbPairs));
probTX_AllAlohaF = np.zeros((numbSim,numbPairs));
# conditional coverage probabilities
probCovCondAllDet = np.zeros((numbSim,numbPairs));
probCovCondAllAlohaA = np.zeros((numbSim,numbPairs));
probCovCondAllAlohaF = np.zeros((numbSim,numbPairs));
# fairness values (found through fitting model with theta)
fairMeanFittedAllDet = np.zeros((numbSim,1));
fairMeanFittedAllAlohaA = np.zeros((numbSim,1));
fairMeanFittedAllAlohaF = np.zeros((numbSim,1));
# optimal theta values
thetaMaxAll = np.zeros((numbSim,numbPairs,numbModelComp));

### START - simulate networks and optimize scheduler START ###
for indexSim in range(numbSim):
    # loop through different network layouts

    # simulate network configuration
    # transmitters
    xxTX = xDelta * (np.random.rand(numbPairs)) + xMin;
    yyTX = yDelta * (np.random.rand(numbPairs)) + yMin;

    # receivers
    if choiceNetwork==1:
        # global model with uniform pairing
        xxRX = 2 * xDelta * (np.random.rand(numbPairs)) + xMin;
        yyRX = 2 * yDelta * (np.random.rand(numbPairs)) + yMin;
    else:
        # local model with local pairing
        thetaRX = 2 * np.pi * np.random.rand(numbPairs);

        # place receiver on desk
        if choiceNetwork==2:
            # random radius
            rhoRX = rMax * np.sqrt(np.random.rand(numbPairs));
        elif choiceNetwork==3:
            # fixed radius
            rhoRX = rMax * np.ones((numbPairs,1));

        # end if
        xxRX_Rel,yyRX_Rel = fun_pol2cart(thetaRX,rhoRX);
        xxRX = xxRX_Rel + xxTX;
        yyRX = yyRX_Rel + yyTX;
    # end if

    # create structure to represent transmitter-receiver point process
    class ppStruct():
        pass
    ppStructConfig = [ppStruct()];
    ppStructConfig[0].xxTX = xxTX;
    ppStructConfig[0].yyTX = yyTX;
    ppStructConfig[0].xxRX = xxRX;
    ppStructConfig[0].yyRX = yyRX;
    ppStructConfig[0].n = numbPairs;
    ppStructConfig[0].window = np.array([xMin,xMax,yMin,yMax]);

    # three different models: determinantal, adaptive Aloha, fixed Aloha
    for indexModel in range(numbModelComp):
        # loop through the three different models
        if indexModel==0:
            # determinantal model
            thetaGuess = np.ones(numbPairs);
            paramKernel_m = paramKernel;
        # end - if
        # Aloha model is equivalent to sigma = 0, so similarity matrix S = I
        if  indexModel==1:
            # Adaptive Aloha
            thetaGuess = np.ones(numbPairs);
            paramKernel_m = np.array(0).flat; #setting sigma = 0
        # end - if
        if  indexModel==2:
            # Fixed Aloha
            thetaGuess = np.array(1).flat;
            paramKernel_m = np.array(0).flat;  #setting sigma = 0
        # end - if

        # calculate similarity matrix S
        S_m = funS(xxTX,yyTX,choiceKernel,paramKernel_m);

        ### START Optimization (ie gradient) method START###
        # define function to be maximized -- see below for funTotalFairDet
        def funMax(theta):
            fairMean,_,_,_ = funFairMeanDet(ppStructConfig,0,\
            thresholdSINR,constNoise,muFading,funPathloss,\
            funFair,S_m,theta,numbFeature);
            return fairMean

        # define function to be minimized
        def funMin(theta):
            return (-1 * funMax(theta))
        # minimize functions -- may take a while
        resultsOpt = minimize(funMin,thetaGuess,options = optionsOpt);
        thetaMax_s_m = resultsOpt.x;
        fairMeanNeg_s_m = resultsOpt.fun;
        numb_theta = len(thetaMax_s_m);
        ### END Optimization END###

        # calculate kernel matrix L
        L_m,qVector_m = funPairsL(xxTX,yyTX,xxRX,yyRX,S_m,thetaMax_s_m,numbFeature);

        # calculate coverage probabilities
        probCov_s_m,probTX_s_m,probCovCond_s_m=\
            funProbCovPairsDetExact(xxTX,yyTX,xxRX,yyRX,\
            thresholdSINR,constNoise,muFading,funPathloss,L_m);

        # collect statistics
        if indexModel==0:
            # determinantal model
            probCovAllDet[indexSim,]=probCov_s_m;
            probTX_AllDet[indexSim,]=probTX_s_m;
            probCovCondAllDet[indexSim,]=probCovCond_s_m;
            fairMeanFittedAllDet[indexSim]=-fairMeanNeg_s_m;
            # for checking purposes
            S_Det = S_m;
            L_Det = L_m;
            qDet = qVector_m;
            thetaMaxDet = thetaMax_s_m;
        # end - if

        if indexModel==1:
            # adaptive Aloha
            probCovAllAlohaA[indexSim,]=probCov_s_m;
            probTX_AllAlohaA[indexSim,]=probTX_s_m;
            probCovCondAllAlohaA[ \
                indexSim,]=probCovCond_s_m;
            fairMeanFittedAllAlohaA[indexSim]=-fairMeanNeg_s_m;
            # for checking purposes
            S_AlohaA = S_m;
            L_AlohaA = L_m;
            qAlohaA = qVector_m;
            thetaMaxAlohaA = thetaMax_s_m;
        # end - if
        if indexModel==2:
            # fixed Aloha
            probCovAllAlohaF[indexSim,]=probCov_s_m;
            probTX_AllAlohaF[indexSim,]=probTX_s_m;
            probCovCondAllAlohaF[indexSim,]=probCovCond_s_m;
            fairMeanFittedAllAlohaF[indexSim]=-fairMeanNeg_s_m;
            # for checking purposes
            S_AlohaF = S_m;
            L_AlohaF = L_m;
            qAlohaF = qVector_m;
            thetaMaxAlohaF = thetaMax_s_m;
        # end - if
    # end - if
# end - loop
### END simulate networks and optimize scheduler END ###

### START calculate ensemble statistics START ###
# coverage probabilities averaged over ensemble (ie all network configurations)
probCovMeanDet = np.mean(probCovAllDet);
probCovMeanAlohaA = np.mean(probCovAllAlohaA);
probCovMeanAlohaF = np.mean(probCovAllAlohaF);
# access probabilities averaged over ensemble
probTXMeanDet = np.mean(probTX_AllDet);
probTXMeanAlohaA = np.mean(probTX_AllAlohaA);
probTXMeanAlohaF = np.mean(probTX_AllAlohaF);
# fairness averaged over each network
fairAllDet=(funFair(probCovAllDet));
fairAllAlohaA=(funFair(probCovAllAlohaA));
fairAllAlohaF=(funFair(probCovAllAlohaF));
# fairness averaged over ensemble
fairMeanDet = np.mean(fairAllDet);
fairMeanAlohaA = np.mean(fairAllAlohaA);
fairMeanAlohaF = np.mean(fairAllAlohaF);
### END calculate ensemble statistics END###

indexPlotExample = numbSim - 1; # choose an index for a plotting example
# retrieve data for plotting
probCovDet_s = probCovAllDet[indexPlotExample,];
probCovAlohaA_s = probCovAllAlohaA[indexPlotExample,];
probCovAlohaF_s = probCovAllAlohaF[indexPlotExample,];
fairDet_s = fairAllDet[indexPlotExample,];
fairAlohaA_s = fairAllAlohaA[indexPlotExample,];
fairAlohaF_s = fairAllAlohaF[indexPlotExample,];

indexPairs = np.arange(numbPairs) + 1; # indices for transmitter-receiver pairs

# loop through different plots
# first plot is coverage probabilities, second plot is fairness values
for indexPlot in range(numbPlot):
    # create figure
    plt.figure();
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams.update({'font.size': 9})
    plt.xticks(indexPairs)

    if indexPlot==0:
        str_yLabel='Coverage probability $P_i$';
        # retrieve plotting data
        dataPlotDet = probCovDet_s;
        dataPlotMeanDet = probCovMeanDet;
        dataPlotAlohaA = probCovAlohaA_s;
        dataPlotMeanAlohaA = probCovMeanAlohaA;
        dataPlotAlohaF = probCovAlohaF_s;
        dataPlotMeanAlohaF = probCovMeanAlohaF;
        # set axis
        plt.axis([1, numbPairs, 0, 1.2]);
    # end - if

    if indexPlot==1:
        str_yLabel=('Fairness');
        # retrieve plotting data
        dataPlotDet = fairDet_s;
        dataPlotMeanDet = fairMeanDet;
        dataPlotAlohaA = fairAlohaA_s;
        dataPlotMeanAlohaA = fairMeanAlohaA;
        dataPlotAlohaF = fairAlohaF_s;
        dataPlotMeanAlohaF = fairMeanAlohaF;
    # end - if

    # plot results
    plt.plot(indexPairs,dataPlotDet,'bx');
    plt.plot(indexPairs,dataPlotAlohaA,'m+');
    plt.plot(indexPairs,dataPlotAlohaF,'k*');

    if boolePlotAverage:
        plt.hlines(dataPlotMeanDet,indexPairs[0],indexPairs[-1],'b',linestyles='solid');
        plt.hlines(dataPlotMeanAlohaA,indexPairs[0],indexPairs[-1],'m',linestyles='dashed');
        plt.hlines(dataPlotMeanAlohaF,indexPairs[0],indexPairs[-1],'k',linestyles='dashdot');

        plt.legend(['Determinantal (Single network)','Determinantal (Ensemble average)',\
            'Adaptive Aloha (Single network)','Adaptive Aloha (Ensemble average)',\
            'Fixed Aloha (Single network)','Fixed Aloha (Ensemble average)']);
    else:
        plt.legend(['Determinantal (Single network)',\
            'Adaptive Aloha (Single network)',\
            'Fixed Aloha (Single network)']);
    # end - if

    plt.xlabel('Pair index i');
    plt.ylabel(str_yLabel);
    plt.title(labelTitle);
