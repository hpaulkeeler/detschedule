# fairMean, fairMeanAll,probCovAll,probAccAll = ...
#    funFairMeanDet(ppStructConfig,indexConfig,...
#    thresholdSINR,constNoise,muFading,funPathloss,funFair,S,theta,numbFeature)
#
# funFairMeanDet calculate the average fairness across an ensemble of
# network configurations.
#
# INPUTS:
#
# ppStructConfig is a list of object class, where each element defines a network
# configuration or point pattern. 
# Here the arrays xxTX,yyTX,xxRX, and yyRX are the Cartesian coordinates
# of the transmitter-receiver n pairs, which are located in a square  with
# dimensions [xMin,xMax,yMin,yMax].
#
# indexConfig is index of the networks configurations being studied.
#
# thresholdSINR is the SINR threshold.
#
# constNoise is the noise constant.
#
# muFading is the mean of the iid random exponential fading variables.
#
# funPathloss is a single - variable function for the path - loss model
# eg funPathloss = lambda r:(((1 + r))^(-3.5));
#
# funFair is a single - variable function for the fairness model
# eg funFair = lambda R : log(R);
#
# OUTPUTS:
#
# fairMean is the fairness averaged over the network ensemble (ie all the
# network configurations).
#
# fairMeanAll is the total fairness for each network configuration.
#
# probCovAll is the coverage probability for each receiver.
#
# probAccAll is the access probability for each transmitter-receiver pair.
#
# This code was originally written by H.P. Keeler for the paper[1], which
# studies determinantal scheduling in wireless networks.
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
# Author: H. Paul Keeler, 2025.

import numpy as np
#from funGeneratePairs import funGeneratePairs
#from funGetConfig import funGetConfig
from funPairsL import funPairsL
from funProbCovPairsDetExact import funProbCovPairsDetExact
import warnings

def funFairMeanDet(ppStructConfig,indexConfig,thresholdSINR,constNoise,muFading,funPathloss,\
                   funFair,S,theta,numbFeature):
    indexConfig = np.array(indexConfig).flat; # convert to flat array access single length ones
    numbConfig = len(indexConfig); # number of configurations sets
    # number of pairs in each configuration
    numbPairsAll = np.array([ppStructConfig[indexConfig[ii]].n for ii in range(numbConfig)])
    indexPairsAll= np.cumsum(numbPairsAll[0:-1]); # index for creating lists
    numbPairsTotal = np.sum(numbPairsAll);# total number of pairs

    # initialize  variable for average fairness in each training set
    fairMeanAll = np.zeros(numbConfig);
    # initialize  list for coverage probability
    probCovAll = np.split((np.zeros((numbPairsTotal))),indexPairsAll)
    # initialize  list for access probability
    probTXAll = np.split((np.zeros((numbPairsTotal))),indexPairsAll)

    # loop through for every training / learning sample
    for tt in range(numbConfig):
        indexConfigTemp = indexConfig[tt];
        # retrieve x / y coordinates of all transmitter-receiver pairs
        xxTX = ppStructConfig[indexConfigTemp].xxTX;
        yyTX = ppStructConfig[indexConfigTemp].yyTX;
        xxRX = ppStructConfig[indexConfigTemp].xxRX;
        yyRX = ppStructConfig[indexConfigTemp].yyTX;

        # create L matrix
        LTemp,_ = funPairsL(xxTX,yyTX,xxRX,yyRX,S,theta,numbFeature);

        # calculate coverage probabilities for all transmitter-receiver pairs
        # probCovTemp,probTXTemp,_ = funProbCovPairsDet(xxTX,yyTX,xxRX,yyRX,fun_h,fun_w,LTemp);
        probCovTemp,probTXTemp,_ = funProbCovPairsDetExact(xxTX,yyTX,xxRX,yyRX,\
            thresholdSINR,constNoise,muFading,funPathloss,LTemp);
        probCovAll[tt]=probCovTemp; # coverage probability
        probTXAll[tt]=probTXTemp; # access probability

        rateTXRX = probCovTemp; # use coverage probability as rate

        # weight for "good" subsets; see funU definition
        weightConfig = 1;
        # calculate mean of fairness of rates
        fairMeanAll[tt]=np.mean(funFair(rateTXRX) * weightConfig);

    fairMean = np.mean(fairMeanAll); # average fairness across all training sets

    # check average fairness
    if (np.isinf(fairMean)):
        warnings.warn('The fairness is infinite, possibly due to exponentially \
                      small coverage probabilities. Adjust network and SINR \
                      parameters.');
    return fairMean,fairMeanAll,probCovAll,probTXAll
