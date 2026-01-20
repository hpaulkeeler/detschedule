# [probCov,probTXRX,probCovCond]=funProbCovTXRXDet(xx,yy,...
#    fun_h,fun_w,L,indexTrans,indexRec)
#
# Calculates the coverage probabilities in a network with
# transmitter-or-receiver nodes based on the signal-to-interference ratio
# (SINR). The network has a random medium access control (MAC) scheme
# based on a determinantal point process, as outlined in the paper[1] by
# B\laszczyszyn, Brochard and Keeler
#
# By coverage, it is assumed that transmitter is active and receiver is not
# active *and* the SINR of the transmitter is larger than some threshold at the
# corresponding receiver.
#
# If you use this code in published research, please cite paper[1].
#
# INPUTS:
#
# xx is the x-coordinate of the nodes.
# yy is the y-coordinate of the nodes.
#
# fun_h is a helper function called the 'interference factor'. The standard
# form is
# fun_h=@(s,r)(1./((funPathloss(s)./funPathloss(r))*thresholdSINR+1));
#
# fun_w is a helper function called the 'noise factor'. The standard
# form is
# fun_w=@(r)(exp(-(thresholdSINR/muFading)*constNoise./funPathloss(r)));
#
# These functions can be created with funFactorFunctions.m  See the paper
# [1] details.
#
# indexTransmit is an (optional) index of the active (ie transmitting and
# receiving pairs). If it doesn't exit, the code assumes all pairs are
# active.
#
# OUTPUTS:
#
# probCov is the coverage probability of all the transmitter-receiver
# pairs.
#
# probTX is the medium access probability, meaning the probability that a
# pair is transmitting and receiving.
#
# probCovCond is the conditional probability that the transmitter
# has a SINR larger than some threshold at receiver.
#
# References:
#
# [1] B\laszczyszyn, Brochard and Keeler, "Coverage probability in
# wireless networks with determinantal scheduling", 2020.
#
# Author: H. Paul Keeler, 2020.

import numpy as np  # NumPy package for arrays, random number generation, etc
from funPalmK import funPalmK  # finding Palm distribution (for a single point)
from funLtoK import funLtoK  # converting L kernel to a (normalized) K kernel


def funProbCovTXRXDet(xx, yy, fun_h, fun_w, L, indexTrans, indexRec):
    #reshape into row vectors
    xx = np.ravel(xx)
    yy = np.ravel(yy)

    #transmitter location
    xxTX = xx[indexTrans]
    yyTX = yy[indexTrans]

    #Receiver location
    xxRX = xx[indexRec]
    yyRX = yy[indexRec]

    ###START Create kernels and Palm kernels START###
    K = funLtoK(L)  # caclulate K kernel from kernel L
    sizeK = K.shape[0]  # number of columns/rows in kernel matrix K

    #Calculate all respective distances (based on random network configuration)
    #from all transmitters to receiver
    dist_ji_xx = np.outer(xx, np.ones((sizeK,))) - \
        np.outer(np.ones((sizeK,)), xxRX)
    dist_ji_yy = np.outer(yy, np.ones((sizeK,))) - \
        np.outer(np.ones((sizeK,)), yyRX)
    dist_ji = np.hypot(dist_ji_xx, dist_ji_yy)  # Euclidean distances
    #transmitters to receivers
    dist_ii_xx = xxTX-xxRX
    dist_ii_yy = yyTX-yyRX
    dist_ii = np.hypot(dist_ii_xx, dist_ii_yy)  # Euclidean distances
    # repeat cols for element-wise evaluation
    dist_ii = np.tile(dist_ii, (sizeK, 1))

    #apply functions
    hMatrix = fun_h(dist_ji, dist_ii)  # matrix H for all h_{x_i}(x_j) values
    W_x = fun_w(np.hypot(xx-xxRX, yy - yyRX))  # noise factor

    ##create h matrix corresponding to transmitter
    booleAll = np.ones(sizeK, dtype=bool)
    booleReduced = booleAll
    booleReduced[indexTrans] = False  # remove transmitter

    #choose transmitter-receiver row
    hVectorReduced = hMatrix[booleReduced, indexTrans]
    #repeat vector hVectorReduced as rows
    hMatrixReduced = np.tile(hVectorReduced, (sizeK-1, 1))
    hMatrixReduced = hMatrixReduced.transpose()

    #create Palm kernels conditioned on transmitter existing
    KPalmReducedTX, KPalmTX = funPalmK(K, indexTrans)
    #create Palm kernels conditioned on receiver existing
    KPalmRXReduced, KPalmRX = funPalmK(K, indexRec)
    #create Palm kernels conditioned on  transmitter AND receiver existing
    _, KPalmTXRX = funPalmK(KPalmTX, indexRec)
    #create reduced (by transmitter) Palm kernel conditioned on transmitter
    #AND receiver existing
    indexReduced = np.arange(sizeK)[booleReduced]
    KPalmSemiReducedTXRX = np.eye(sizeK-1)
    for i in range(KPalmTXRX.shape[0]-1):
        KPalmSemiReducedTXRX[:, i] = KPalmTXRX[indexReduced, indexReduced[i]]

    #calculate final kernels
    #for transmitter
    KReduced_hTX = np.sqrt(1-hMatrixReduced.transpose()) * \
        KPalmReducedTX*np.sqrt(1-hMatrixReduced)
    ##for reciever and transmitter
    KReduced_hRX = np.sqrt(1-hMatrixReduced.transpose()) * \
        KPalmSemiReducedTXRX*np.sqrt(1-hMatrixReduced)
    ###END Create kernels and Palm kernels END###

    ###START Connection Proability (ie SINR>thresholdConst) START###
    #calculate probabiliity for the event that transmitter's
    #signal at the receiver has an SINR>thresholdConst, given the pair is
    # active (ie trasnmitting and receiving); see Section IV in paper[1].

    #probability transmitter exists (ie transmitter at indexTrans) - event B
    probB = K[indexTrans, indexTrans]

    #probability transmitter but no receiver
    indexPair = np.array([indexTrans, indexRec])
    probBNotC = np.linalg.det(K[indexPair, :][:, indexPair])

    #probability transmitter and receiver existing
    probBandC = probB-probBNotC

    #probability of SINR>threshold (ie transmiter is connected ) given B
    probA_GivenB = np.linalg.det(np.eye(sizeK-1)-KReduced_hTX)*W_x[indexTrans]

    #probability of SINR>threshold (ie transmiter is connected ) given B and C
    probA_GivenBNotC = np.linalg.det(
        np.eye(sizeK-1)-KReduced_hRX)*W_x[indexTrans]

    #probability NOT C (ie a transmitter exists at indexRec) given B
    probNotC_GivenB = KPalmTX[indexRec, indexRec]

    #probability C given B
    probC_GivenB = 1-probNotC_GivenB

    #coverage probability ie probability of A given B and C
    probA_GivenBandC = (probA_GivenB-probNotC_GivenB *
                        probA_GivenBNotC)/probC_GivenB

    probCovCond = probA_GivenBandC  # conditional coverage probability
    probTXRX = probBandC  # probability of pair existing
    #connection probability
    probCov = probCovCond*probTXRX
    ###END Connection Proability (ie SINR>thresholdConst) END###

    return probCov, probTXRX, probCovCond
