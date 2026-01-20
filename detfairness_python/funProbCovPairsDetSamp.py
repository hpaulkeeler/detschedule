# [probCovEmp,probTXEmp,probCovCondEmp]=funProbCovPairsDetSamp(xxTX,yyTX,xxRX,yyRX,...
#    thresholdSINR,constNoise,muFading,funPathloss,L,numbSamp,indexTransPair)
#
# Using random simulations, the code estimates the coverage
# probabilities in a bi-pole (or bipolar) network of transmitter-receiver
# pairs based on the signal-to-interference-plus-noise ratio (SINR). The
# network has a random medium access control (MAC) scheme based on a
# determinantal point process, as outlined in the paper[1] by Blaszczyszyn
# and Keeler or the paper[2] by Blaszczyszyn, Brochard and Keeler.
#
# By coverage, it is assumed that each transmitter-receiver pair is active
# *and* the SINR of the transmitter is larger than some threshold at the
# corresponding receiver.
#
# If you use this code in published research, please cite paper[1].
#
# INPUTS:
#
# xxTX, yyTX are vectors for the Carestian coordinates of the transmitters.
# xxRX, yyRX are vectors for the Carestian coordinates of the receivers.
#
# thresholdSINR is the SINR threshold.
#
# constNoise is the noise constant.
#
# muFading is the mean of the iid random exponential fading variables.
#
# funPathloss is a single - variable function for the path - loss model
# eg funPathloss = @(r)(((1 + r)).^(-3.5));
#
# L is the L - ensemble kernel, which can be created using the file
# funPairsL.m
#
# indexTransmit is an (optional) index of the active (ie transmitting and
# receiving pairs). If it doesn't exist, the code assumes all pairs are
# active.
#
# OUTPUTS:
#
# probCov is the coverage probability of all the transmitter - receiver
# pairs.
#
# probTX is the medium access probability, meaning the probability that a
# pair is transmitting and receiving.
#
# probCovCond is the conditional probability that the transmitter
# has a SINR larger than some threshold at receiver.
#
# This code was used by H.P. Keeler for the paper[1] by
# Blaszczyszyn and Keeler, which studies determinantal scheduling in wireless
# networks.
#
#
# If you use this code in published research, please cite paper[1].
#
# References:
#
# [1] Blaszczyszyn and Keeler, "Adaptive determinantal scheduling with
# fairness in wireless networks", 2025.
#
# [2] Blaszczyszyn, Brochard and Keeler, "Coverage probability in
# wireless networks with determinantal scheduling", 2020.
#
# Author: H. Paul Keeler, 2025.

import numpy as np; # NumPy package for arrays, random number generation, etc
from funSimSimpleDPP import funSimSimpleDPP;

def  funProbCovPairsDetSamp(xxTX,yyTX,xxRX,yyRX,thresholdSINR,constNoise,\
    muFading,funPathloss,L,numbSamp,indexTransPair=-1):
    
    # eigen decomposition
    eigenValL, eigenVecL = np.linalg.eig(L)

    # reshape into row vectors
    xxTX = np.ravel(xxTX);
    yyTX = np.ravel(yyTX);
    xxRX = np.ravel(xxRX);
    yyRX = np.ravel(yyRX);

    sizeL = L.shape[0]; # number of columns / rows in kernel matrix K

    numbPairs = sizeL;

    if indexTransPair==-1:
        indexTransPair=np.arange(sizeK); # set index for all pairs
    else:
        indexTransPair = np.ravel(indexTransPair); # used supplied index

    # initiate vectors for probability
    probTXEmp = np.zeros(len(indexTransPair)); 
    probCovCondEmp = np.zeros(len(indexTransPair)); 
    probCovEmp = np.zeros(len(indexTransPair)); 

    # loop through the network pairs
    for pp in range(len(indexTransPair)):
        indexTransTemp = indexTransPair[pp]; # index of current pair

        # transmitter location
        xxTX0 = xxTX[indexTransTemp];
        yyTX0 = yyTX[indexTransTemp];

        # receiver location
        xxRX0 = xxRX[indexTransTemp];
        yyRX0 = yyRX[indexTransTemp];

        ### START Empirical Connection Probability (ie SINR>thresholdConst) START###
        # initialize  boolean vectors / arrays for collecting statistics
        booleTX = np.zeros(numbSamp,dtype = bool); # transmitter-receiver pair exists
        booleCov = np.zeros(numbSamp,dtype = bool); # transmitter-receiver pair is connected
        
        # loop through all simulations
        for ss in range(numbSamp):
            indexDPP = funSimSimpleDPP(eigenVecL, eigenValL)
            #if transmitter-receiver pair exists in determinantal outcome
            booleTX[ss] = any(indexDPP == indexTransPair)

            if booleTX[ss]:
                #create Boolean variable for active interferers
                booleInter = np.zeros(numbPairs, dtype = bool)
                booleInter[indexDPP] = True
                booleInter[indexTransPair] = False

                #x / y values of interfering nodes
                xxInter = xxTX[booleInter]
                yyInter = yyTX[booleInter]

                #number of interferers
                numbInter = np.sum(booleInter)

                #simulate signal for interferers
                fadeRandInter = np.random.exponential(muFading, numbInter)  # fading
                distPathInter = np.hypot(xxInter - xxRX0, yyInter - yyRX0)  # path distance
                proplossInter = fadeRandInter * funPathloss(distPathInter)  # pathloss

                #simulate signal for transmitter
                fadeRandSig = np.random.exponential(muFading)  # fading
                distPathSig = np.hypot(xxTX0 - xxRX0, yyTX0 - yyRX0)  # path distance
                proplossSig = fadeRandSig * funPathloss(distPathSig)  # pathloss

                #Calculate the SINR
                SINR = proplossSig / (np.sum(proplossInter) + constNoise)

                #see if transmitter is connected
                booleCov[ss] = (SINR > thresholdSINR)
                
        # Estimate empirical probabilities
        probCovCondEmp[pp] = np.mean(booleCov[booleTX]);# SINR>thresholdConst given pair
        probTXEmp[pp] = np.mean(booleTX); # transmitter-receiver pair exists
        probCovEmp[pp] = np.mean(booleCov); # SINR>thresholdConst
    ### END Empirical Connection Probability (ie SINR>thresholdConst) END###
    return probCovEmp, probTXEmp, probCovCondEmp
            
