# Simulates a network with nodes, where each node can be either a
# transmitter or receiver (but not both) at any time step. The simulation
# examines the coverage based on the signal-to-interference ratio (SINR).
# The network has a random medium access control (MAC) scheme based on a
# determinantal point process, as outlined in the paper[1] by
# B\laszczyszyn, Brochard and Keeler. This code validates by simulation
# Propositions IV.1 and IV.2 in the paper[1]. This result gives the
# probability of coverage based on the SINR value of a transmitter-receiver
# pair in a non-random network of transmitter-or-receiver nodes such as a
# realization of a random point process.
#
# More specifically, the code estimates the probability of x and y being
# connected (ie SINR(x,y)>tau)given that x is transmitting and
# y isn't.
#
# The simulation section estimates the empirical probability of SINR-based
# coverage. For a large enough number of simulations, this empirical result
# will agree with the analytic results given in the paper[2].
#
# By coverage, it is assumed that the SINR of the transmitter is larger
# than some threshold at the corresponding receiver.
#
# Probabilities for other events are calculated/estimated including:
#
# Event A=SINR(x,y) > tau
# Event B=Transmitter exists
# Event C=Receiver exists
#
# This code was originally written by H.P Keeler for the paper by
# B\laszczyszyn, Brochard and Keeler[1].
#
# If you use this code in published research, please cite paper[1].
#
# References:
#
# [1] B\laszczyszyn, Brochard and Keeler, "Coverage probability in
# wireless networks with determinantal scheduling", 2020.
#
# Author: H. Paul Keeler, 2020.

from funProbCovTXRXDet import funProbCovTXRXDet
import numpy as np  # NumPy package for arrays, random number generation, etc
import matplotlib.pyplot as plt  # for plotting

# simulate determintal point process
from funSimSimpleDPP import funSimSimpleDPP
from funPalmK import funPalmK  # find Palm distribution (for a single point)
from funLtoK import funLtoK  # convert L kernel to a (normalized) K kernel

plt.close("all")  # close all figures

#set random seed for reproducibility
np.random.seed(1)

###START -- Parameters -- START###
choiceExample = 1  # 1 or 2 for a random (uniform) or deterministic example

numbSim = 10**4  # number of simulations
numbNodes = 10  # number of pairs
indexTrans = 0  # index for transmitter
indexRec = 1  # index for receiver
#above indices are bounded by numbNodes

#fading model
muFading = 1/3  # Rayleigh fading average
#path loss model
betaPath = 2  # pathloss exponent
kappaPath = 1  # rescaling constant for pathloss function

thresholdSINR = 0.1  # SINR threshold value
constNoise = 0  # noise constant

#Determinantal kernel parameters
choiceKernel = 1  # 1 for Gaussian (ie squared exponetial );2 for Cauchy
#3 for independent (ie binomial) model
sigma = 1  # parameter for Gaussian and Cauchy kernel
alpha = 1  # parameter for Cauchy kernel
pAloha = 0.5  # parameter for independent kernel (ie proportion transmitting)

#Simulation window parameters
xMin = -1
xMax = 1  # x dimensions
yMin = -1
yMax = 1  # y dimensions
xDelta = xMax-xMin  # rectangle width
yDelta = yMax-yMin  # rectangle height
###END -- Parameters -- END###

#Simulate a random point process for the network configuration
#interferer section
if (choiceExample == 1):
    #random (uniform) x/y coordinates
    #transmitters or receivers
    xx = xDelta*(np.random.rand(numbNodes))+xMin
    yy = yDelta*(np.random.rand(numbNodes))+yMin

else:
    #non-random x/y coordinates
    #transmitters or receivers
    t = 2*np.pi*np.linspace(0, (numbNodes-1)/numbNodes, numbNodes)
    xx = (1+np.cos(5*t+1))/2
    yy = (1+np.sin(3*t+2))/2

#transmitter location
xxTX = xx[indexTrans]
yyTX = yy[indexTrans]

#Receiver location
xxRX = xx[indexRec]
yyRX = yy[indexRec]

# START -- CREATE L matrix -- START
sizeL = numbNodes
#Calculate Gaussian or Cauchy kernel based on grid x/y values
#all squared distances of x/y difference pairs
xxDiff = np.outer(xx, np.ones((sizeL,)))-np.outer(np.ones((sizeL,)), xx)
yyDiff = np.outer(yy, np.ones((sizeL,)))-np.outer(np.ones((sizeL,)), yy)
rrDiffSquared = (xxDiff**2+yyDiff**2)

if choiceKernel == 1:
    #Gaussian/squared exponential kernel
    L = np.exp(-(rrDiffSquared)/sigma**2)

elif choiceKernel == 2:
    #Cauchy kernel
    L = 1/(1+rrDiffSquared/sigma**2)**(alpha+1/2)

else:
    raise Exception('choiceKernel has to be equal to 1 or 2.')

L = 10*L  # scale matrix up (increases the eigenvalues ie number of points)
# END-- CREATE L matrix -- # END

#Eigen decomposition
eigenValL, eigenVecL = np.linalg.eig(L)

#Helper functions


def funPathloss(r):
    return (kappaPath*(1+r))**(-betaPath)  # pathloss function
#Functions for the proability of being connected


def fun_h(s, r):
    return (1/(thresholdSINR*(funPathloss(s)/funPathloss(r))+1))


def fun_w(r):
 return (np.exp(-(thresholdSINR/muFading)*constNoise/funPathloss(r)))


#initialize  boolean vectors/arrays for collecting statistics
booleA = np.zeros(numbSim, dtype=bool)  # transmitter is connected
booleB = np.zeros(numbSim, dtype=bool)  # transmitter exists
booleC = np.zeros(numbSim, dtype=bool)  # receiver exists
#loop through all simulations
for ss in range(numbSim):
    #DPP for active transmitter nodes
    indexDPP = funSimSimpleDPP(eigenVecL, eigenValL)

    booleB[ss] = any(indexDPP == indexTrans)  # if transmitter is in subset
    booleC[ss] = all(indexDPP != indexRec)  # if receiver is not in subset

    #if transmitter is in the determinantal subset, calculate its SINR
    if booleB[ss]:
        #create Boolean variable for active interferers
        booleInter = np.zeros(numbNodes, dtype=bool)
        booleInter[indexDPP] = True
        booleInter[indexTrans] = False  # exclude transmitter

        #x/y values of interfering nodes
        xxInter = xx[booleInter]
        yyInter = yy[booleInter]

        #number of interferers
        numbInter = np.sum(booleInter)

        #simulate signal for interferers
        fadeRandInter = np.random.exponential(muFading, numbInter)  # fading
        distPathInter = np.hypot(xxInter-xxRX, yyInter-yyRX)  # path distance
        proplossInter = fadeRandInter*funPathloss(distPathInter)  # pathloss

        #simulate signal for transmitter
        fadeRandSig = np.random.exponential(muFading)  # fading
        distPathSig = np.hypot(xxTX-xxRX, yyTX-yyRX)  # path distance
        proplossSig = fadeRandSig*funPathloss(distPathSig)  # pathloss

        #Calculate the SINR
        SINR = proplossSig/(np.sum(proplossInter)+constNoise)

        #see if transmitter is connected
        booleA[ss] = (SINR > thresholdSINR)


booleBandC = booleB & booleC  # transmitter-receiver pair exists
booleNotC = ~booleC  # receiver does not exist
booleBandNotC = booleB & booleNotC  # transmitter exists, receiver does not

###START Create kernels and Palm kernels START###
K = funLtoK(L)  # caclulate K kernel from kernel L
sizeK = K.shape[0]  # number of columns/rows in kernel matrix K

#Calculate all respective distances (based on random network configuration)
#from all transmitters to receiver
dist_ji_xx = np.outer(xx, np.ones((sizeK,)))-np.outer(np.ones((sizeK,)), xxRX)
dist_ji_yy = np.outer(yy, np.ones((sizeK,)))-np.outer(np.ones((sizeK,)), yyRX)
dist_ji = np.hypot(dist_ji_xx, dist_ji_yy)  # Euclidean distances
#transmitters to receivers
dist_ii_xx = xxTX-xxRX
dist_ii_yy = yyTX-yyRX
dist_ii = np.hypot(dist_ii_xx, dist_ii_yy)  # Euclidean distances
# repeat cols for element-wise evaluation
dist_ii = np.tile(dist_ii, (sizeK, 1))

#apply functions
hMatrix = fun_h(dist_ji, dist_ii)  # matrix H for all h_{x_i}(x_j) values
W_x = fun_w(np.hypot(xx-xxRX, yy-yyRX))  # noise factor

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
probB_Emp = np.mean(booleB)

#probability receiver exists (ie no transmitter at indexRec) - event C
probC = 1-K[indexRec, indexRec]
probC_Emp = np.mean(booleC)

#probability transmitter but no receiver
indexPair = np.array([indexTrans, indexRec])
probBNotC = np.linalg.det(K[indexPair, :][:, indexPair])
probBNotC_Emp = np.mean(booleBandNotC)
#
#probability transmitter and receiver existing
probBandC = probB-probBNotC
probBandC_Emp = np.mean(booleBandC)

#probability of SINR>threshold (ie transmiter is connected ) given B
probA_GivenB = np.linalg.det(np.eye(sizeK-1)-KReduced_hTX)*W_x[indexTrans]
probA_GivenB_Emp = np.mean(booleA[booleB])

#probability of SINR>threshold (ie transmiter is connected ) given B and C
probA_GivenBNotC = np.linalg.det(np.eye(sizeK-1)-KReduced_hRX)*W_x[indexTrans]
probA_GivenBNotC_Emp = np.mean(booleA[booleNotC])

#probability B given NOT C (ie a transmitter exists at indexRec)
probB_GivenNotC = KPalmRX[indexTrans, indexTrans]
probB_GivenNotC_Emp = np.mean(booleB[booleNotC])

#probability B given C
probB_GivenC = (probB-(1-probC)*probB_GivenNotC)/probC
probB_GivenC_Emp = np.mean(booleB[booleC])

#probability NOT C (ie a transmitter exists at indexRec) given B
probNotC_GivenB = KPalmTX[indexRec, indexRec]
probNotC_GivenB_Emp = np.mean(booleNotC[booleB])

#probability C given B
probC_GivenB_Emp = np.mean(booleC[booleB])
probC_GivenB = 1-probNotC_GivenB

print('Conditional coverage probability (ie A given B and C).')
#coverage probability ie probability of A given B and C
probA_GivenBandC = (probA_GivenB-probNotC_GivenB*probA_GivenBNotC)/probC_GivenB
print('probA_GivenBandC = ', probA_GivenBandC)

#Estimate empirical probability two different ways
#Directly
probA_GivenBandC_Emp1 = np.mean(booleA[booleBandC])
print('probA_GivenBandC_Emp1 = ', probA_GivenBandC_Emp1)

#Indirectly
probA_GivenBandC_Emp2 = (probA_GivenB_Emp-probNotC_GivenB_Emp*probA_GivenBNotC_Emp)\
    / probC_GivenB_Emp

print('Coverage probability (ie A given B and C).')
#connection probability
probCov = probA_GivenBandC*probBandC
print('probCov = ', probCov)
probCov_Emp1 = np.mean(booleA & booleB & booleC)
print('probCov_Emp1 = ', probCov_Emp1)
#probCov_Emp2=probA_GivenBandC_Emp2*probBandC_Emp

#probCovCond=probA_GivenBandC #conditional coverage probability
#probTXRX=probBandC #probability of pair existing
#connection probability
#probCov=probCovCond*probTXRX

###END Connection Proability (ie SINR>thresholdConst) END###

#TEST
probCov, probTXRX, probCovCond = funProbCovTXRXDet(
    xx, yy, fun_h, fun_w, L, indexTrans, indexRec)

if indexDPP.size > 0:
    ### START -- Plotting -- START ###
    markerSize = 13
    #random color vector
    vectorColor = np.random.rand(3)  # random vector for colors of marker
    #Plot point process
    plt.plot(xx, yy, 'ko', markerfacecolor="None", markersize=markerSize)
    #Plot determinantally-thinned point process
    plt.plot(xx[indexDPP], yy[indexDPP], 'k.', markerfacecolor=vectorColor,
             markersize=1.1*markerSize, markeredgecolor='none')
    plt.axis('equal')
    plt.axis('off')
    plt.legend(('Original point process', 'Determinantal subset'))
    ### END -- Plotting -- END ###
#end
