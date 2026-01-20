# [probCov,probTX,probCovCond]=funProbCovDet(xxTX,yyTX,xxRX,yyRX,...
#    fun_h,fun_w,L)
# Calculates the coverage probabilities in a network with
# transmitter-receiver pairs based on the signal-to-interference ratio
# (SINR). The network has a random medium access control (MAC) scheme
# based on a determinantal point process, as outlined in the paper[1] by
# B\laszczyszyn, Brochard and Keeler
#
# By coverage, it is assumed that each transmitter-receiver pair is active
# *and* the SINR of the transmitter is larger than some threshold at the
# corresponding receiver.
#
# If you use this code in published research, please cite paper[1].
#
# INPUTS:
#
# xxTX is the x-coordinate of the transmitters.
# yyTX is the y-coordinate of the transmitters.
# xxRX is the x-coordinate of the receivers.
# yyRX is the y-coordinate of the receivers.
#
# fun_h is a helper function called the 'interference factor'. The standard
# form is
# fun_h=(1-p/(((s/r)^betaPath)/thresholdSINR+1))
#
# fun_w is a helper function called the 'noise factor'. The standard
# form is
# fun_w=(exp(-thresholdSINR*(constNoise/muFading)*(kappaPath*s)^betaPath));
# These functions can be created with funFactorFunctions.py  See the papers
# [1] and [2] for details.
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
# [1] B\laszczyszyn, Brochard and Keeler INSERT PAPER
#
# Author: H. Paul Keeler, 2019.

import numpy as np  # NumPy package for arrays, random number generation, etc
from funPalmK import funPalmK  # finding Palm distribution (for a single point)
from funLtoK import funLtoK  # converting L kernel to a (normalized) K kernel


def funProbCovPairsDet(xxTX, yyTX, xxRX, yyRX, fun_h, fun_w, L):
    #reshape into row vectors
    xxTX = np.ravel(xxTX)
    yyTX = np.ravel(yyTX)
    xxRX = np.ravel(xxRX)
    yyRX = np.ravel(yyRX)

    ###START Numerical Connection Proability (ie SINR>thresholdSINR) START###
    K = funLtoK(L)  # caclulate K kernel from kernel L
    sizeK = K.shape[0]  # number of columns/rows in kernel matrix K

    #calculate all respective distances (based on random network configuration)
    #transmitters to other receivers
    dist_ji_xx = np.outer(xxTX, np.ones((sizeK,))) - \
        np.outer(np.ones((sizeK,)), xxRX)
    dist_ji_yy = np.outer(yyTX, np.ones((sizeK,))) - \
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
    W_x = fun_w(np.hypot(xxTX-xxRX, yyTX - yyRX))  # noise factor

    probTX = np.diag(K)  # transmitting probabilities are diagonals of kernel
    probCovCond = np.zeros(sizeK)  # intitiate vector forcoverage probability

    #Loop through for all pairs
    for pp in range(sizeK):
        indexTransPair = pp  # index of current pair

        #create h matrix corresponding to transmitter-receiver pair
        # Boolean vector for all pairs
        booleReduced = np.ones(sizeK, dtype=bool)
        booleReduced[indexTransPair] = False  # remove transmitter
        #choose transmitter-receiver row
        hVectorReduced = hMatrix[booleReduced, indexTransPair]
        #repeat vector hVectorReduced as rows
        hMatrixReduced = np.tile(hVectorReduced, (sizeK-1, 1))
        hMatrixReduced = hMatrixReduced.transpose()

        #create reduced Palm kernels
        # reduced Palm version of K matrix
        KPalmReduced, _ = funPalmK(K, indexTransPair)
        #calculate kernel
        KReduced_h = np.sqrt(1-hMatrixReduced.transpose())*KPalmReduced\
            * np.sqrt(1-hMatrixReduced)

        #calculate unconditional probabiliity for the event that transmitter's
        #signal at the receiver has an SINR>tau, given the pair is active (ie
        #trasnmitting and receiving); see equation (??) in [2]
        probCovCond[pp] = np.linalg.det(
            np.eye(sizeK-1)-KReduced_h)*W_x[indexTransPair]

    #calculate unconditional probability
    probCov = probTX*probCovCond
    ###END Numerical Connection Proability START###
    return probCov, probTX, probCovCond
