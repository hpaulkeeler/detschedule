# probCov,probTX,probCovCond = funProbCovPairsDetExact(xxTX,yyTX,...,
#    xxRX,yyRX,thresholdSINR,constNoise,muFading,funPathloss,L,indexTransPair)
#
# Using closed-form expressions, the code calculates the coverage 
# probabilities in a bi-bole (or bipolar) network of transmitter-receiver 
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
# eg funPathloss=lambda r: (((1 + r))^(-3.5))
#
# L is the L - ensemble kernel, which can be created using the file
# funPairsL.m
#
# indexTransPair is an (optional) index for which the probabilities are 
# obtained. If the variable isn't passed, the code calculuates the probabilities
#  for all the transmitter-receiver pairs.
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
# This code is a modification of that funProbCovPairsDet.m used
# originally for paper[2] by Blaszczyszyn, Brochard and Keeler.
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
from funPalmK import funPalmK # finding Palm distribution (for a single point)
from funLtoK import funLtoK # converting L kernel to a (normalized) K kernel

def  funProbCovPairsDetExact(xxTX,yyTX,xxRX,yyRX,thresholdSINR,constNoise,\
    muFading,funPathloss,L,indexTransPair=-1):
    # reshape into row vectors
    xxTX = np.ravel(xxTX);
    yyTX = np.ravel(yyTX);
    xxRX = np.ravel(xxRX);
    yyRX = np.ravel(yyRX);
    
    # helper functions for the probability of being connected
    def fun_h(s,r):
        return (1 / (thresholdSINR * (funPathloss(s) / funPathloss(r)) + 1));
    def fun_w(r):
        return (np.exp(-(thresholdSINR / muFading) * constNoise / funPathloss(r)));

    ### START Numerical Connection Probability (ie SINR>thresholdSINR) START###
    K = funLtoK(L); # calculate K kernel from kernel L
    sizeK = K.shape[0]; # number of columns / rows in kernel matrix K
    
    # calculate all respective distances (based on random network configuration)
    # transmitters to other receivers
    dist_ji_xx = np.outer(xxTX, np.ones((sizeK,))) - np.outer( np.ones((sizeK,)),xxRX);
    dist_ji_yy = np.outer(yyTX, np.ones((sizeK,))) - np.outer( np.ones((sizeK,)),yyRX)
    dist_ji = np.hypot(dist_ji_xx,dist_ji_yy); # Euclidean distances
    # transmitters to receivers
    dist_ii_xx = xxTX - xxRX;
    dist_ii_yy = yyTX - yyRX;
    dist_ii = np.hypot(dist_ii_xx,dist_ii_yy); # Euclidean distances
    dist_ii = np.tile(dist_ii,(sizeK,1));# repeat cols for element - wise evaluation
    
    # apply functions
    hMatrix = fun_h(dist_ji,dist_ii); # matrix H for all h_{x_i}(x_j) values
    W_x = fun_w(np.hypot(xxTX - xxRX,yyTX- yyRX)); # noise factor
    
    if indexTransPair==-1:
        indexTransPair=np.arange(sizeK); # set index for all pairs
    else:
        indexTransPair = np.ravel(indexTransPair); # used supplied index

    probTX = (np.diag(K))[indexTransPair];# transmitting probabilities are diagonals of kernel 
    probCovCond = np.zeros(len(indexTransPair)); # coverage probability       
    
    # Loop through for all pairs
    for pp in range(len(indexTransPair)):
        indexTransPair = pp; # index of current pair
        
        # create h matrix corresponding to transmitter-receiver pair
        booleReduced = np.ones(sizeK,dtype = bool); # Boolean vector for all pairs
        booleReduced[indexTransPair]=False;# remove transmitter
        # choose transmitter-receiver row
        hVectorReduced = hMatrix[booleReduced,indexTransPair]; 
        # repeat vector hVectorReduced as rows
        hMatrixReduced = np.tile(hVectorReduced,(sizeK - 1,1))
        hMatrixReduced = hMatrixReduced.transpose();
        
        # create reduced Palm kernels
        KPalmReduced,_ = funPalmK(K,indexTransPair); # reduced Palm version of K matrix
        # calculate kernel
        KReduced_h = np.sqrt(1 - hMatrixReduced.transpose()) * KPalmReduced\
        *np.sqrt(1 - hMatrixReduced);
        
        # calculate unconditional probability for the event that transmitter's
        # signal at the receiver has an SINR>tau, given the pair is active (ie
        # transmitting and receiving)
        probCovCond[pp]=np.linalg.det(np.eye(sizeK - 1) - KReduced_h) * W_x[indexTransPair];
                  
    # calculate unconditional probability
    probCov = probTX * probCovCond;     
    ### END Numerical Connection Probability START###
    return probCov,probTX,probCovCond
