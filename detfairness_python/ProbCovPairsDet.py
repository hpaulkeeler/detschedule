# Simulates a network with transmitter-receiver pairs in order to examine
# the coverage based on the signal-to-interference-plus-ratio ratio (SINR).  
# The bi-pole network has a random medium access control (MAC) scheme based on a
# determinantal point process, as outlined in the paper[1] by
# Blaszczyszyn, Brochard and Keeler. This code validates by simulation
# Propositions III.1 and III.2 in the paper[1]. These results give the 
# probability of coverage based on the SINR value of a transmitter-receiver
# pair in a non - random network of transmitter-receiver pairs such as a 
# realization of a random point process.
#
# The simulation section estimates the empirical probability of SINR-based
# coverage. For a large enough number of simulations, this empirical result
# will agree with the analytic result given by Propositions III.1 and III.2
# in the paper[1].
#
# By coverage, it is assumed that the transmitter-receiver pair are active
# *and* the SINR of the transmitter is larger than some threshold at the
# corresponding receiver.
#
# It was originally written by H.P. Keeler for the paper[2] by Blaszczyszyn, 
# Brochard and Keeler.
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


from funProbCovPairsDetExact import funProbCovPairsDetExact
import numpy as np  # NumPy package for arrays, random number generation, etc
from funSimSimpleDPP import funSimSimpleDPP
# simulate determintal point process
from funPalmK import funPalmK  # find Palm distribution (for a single point)
from funLtoK import funLtoK  # convert L kernel to a (normalized) K kernel

#set random seed for reproducibility
np.random.seed(1)

### START -- Parameters -- START###
# configuration choice
choiceExample = 2; # 1 or 2 for a random (uniform) or deterministic example
numbSim = 10**4; # number of simulations
numbPairs = 5;# number of pairs
indexTransPair = 0; # index for transmitter-receiver pair
# the above index has to be such that 1<=indexTransPair<=numbPairs

# fading model
muFading = 1 / 3; # Rayleigh fading average
# path loss model
betaPath = 2; # pathloss exponent
kappaPath = 1; # rescaling constant for pathloss function
# SINR model
thresholdSINR = 0.1; # SINR threshold value
constNoise = 0; # noise constant

# choose kernel
choiceKernel = 1; # 1 for Gaussian;2 for Cauchy
sigma = 0;# parameter for Gaussian and Cauchy kernel
alpha = 1;# parameter for Cauchy kernel

# simulation window parameters
# x / y dimensions
xMin = -1;
xMax = 1;
yMin = -1;
yMax = 1;
# width / height
xDelta = xMax - xMin;
yDelta = yMax - yMin;
# x / y centre of window
### END -- Parameters -- END###

#Simulate a random point process for the network configuration
#interferer section
if (choiceExample == 1):
    #random (uniform) x / y coordinates
    #transmitters
    xxTX = xDelta * (np.random.rand(numbPairs)) + xMin
    yyTX = yDelta * (np.random.rand(numbPairs)) + yMin
    #receivers
    xxRX = xDelta * (np.random.rand(numbPairs)) + xMin
    yyRX = yDelta * (np.random.rand(numbPairs)) + yMin
else:
    #non - random x / y coordinates
    #transmitters
    t = 2 * np.pi * np.linspace(0, (numbPairs - 1) / numbPairs, numbPairs)
    xxTX = (1 + np.cos(5 * t+1)) / 2
    yyTX = (1 + np.sin(3 * t+2)) / 2
    #receivers
    xxRX = (1 + np.sin(3 * t+1)) / 2
    yyRX = (1 + np.cos(7 * t+2)) / 2

#transmitter location
xxTX0 = xxTX[indexTransPair]
yyTX0 = yyTX[indexTransPair]

#Receiver location
xxRX0 = xxRX[indexTransPair]
yyRX0 = yyRX[indexTransPair]

# START -- CREATE L matrix -- START
sizeL = numbPairs
xx = xxTX
yy = yyTX
#Calculate Gaussian or Cauchy kernel based on grid x / y values
#all squared distances of x / y difference pairs
xxDiff = np.outer(xx, np.ones((sizeL,))) - np.outer(np.ones((sizeL,)), xx)
yyDiff = np.outer(yy, np.ones((sizeL,))) - np.outer(np.ones((sizeL,)), yy)
rrDiffSquared = (xxDiff**2 + yyDiff**2)

if sigma!=0:
    if choiceKernel == 1:
        # Gaussian / squared exponential kernel
        L = np.exp(-(rrDiffSquared) / sigma**2)

    elif choiceKernel == 2:
        # Cauchy kernel
        L = 1 / (1 + rrDiffSquared / sigma**2)**(alpha + 1/2)

    else:
        raise Exception('choiceKernel has to be equal to 1 or 2.')
else:
    # use identity matrix, which is the Aloha model
    L = np.eye(numbPairs);

L = 10 * L  # scale matrix up (increases the eigenvalues ie number of points)
# END-- CREATE L matrix -- # END

# eigen decomposition
eigenValuesL, eigenVectorsL = np.linalg.eig(L)

# helper functions
def funPathloss(r):
    return (kappaPath * (1 + r)) ** (-betaPath)  # pathloss function
# functions for the proability of being connected
def fun_h(s, r):
    return (1 / (thresholdSINR * (funPathloss(s) / funPathloss(r)) + 1))

def fun_w(r):
 return (np.exp(-(thresholdSINR / muFading) * constNoise / funPathloss(r)))


###START Empirical Connection Probability (ie SINR>thresholdConst) START###
#initialize  boolean vectors / arrays for collecting statistics
booleTX = np.zeros(numbSim, dtype = bool)  # transmitter-receiver pair exists
# transmitter-receiver pair is connected
booleCov = np.zeros(numbSim, dtype = bool)
#loop through all simulations
for ss in range(numbSim):
    indexDPP = funSimSimpleDPP(eigenVectorsL, eigenValuesL)
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

#Estimate empirical probabilities
probCovCondEmp = np.mean(booleCov[booleTX])  # SINR>thresholdConst given pair
probTXEmp = np.mean(booleTX)  # transmitter-receiver pair exists
probCovEmp = np.mean(booleCov)  # SINR>thresholdConst


###START Numerical Connection Probability (ie SINR>thresholdSINR) START###
K = funLtoK(L)  # calculate K kernel from kernel L
sizeK = K.shape[0]  # number of columns / rows in kernel matrix K

#calculate all respective distances (based on random network configuration)
#transmitters to other receivers
dist_ji_xx = np.outer(xxTX, np.ones((sizeL,))) - \
    np.outer(np.ones((sizeL,)), xxRX)
dist_ji_yy = np.outer(yyTX, np.ones((sizeL,))) - \
    np.outer(np.ones((sizeL,)), yyRX)
dist_ji = np.hypot(dist_ji_xx, dist_ji_yy)  # Euclidean distances
#transmitters to receivers
dist_ii_xx = xxTX - xxRX
dist_ii_yy = yyTX - yyRX
dist_ii = np.hypot(dist_ii_xx, dist_ii_yy)  # Euclidean distances
# repeat cols for element - wise evaluation
dist_ii = np.tile(dist_ii, (sizeL, 1))

#apply functions
hMatrix = fun_h(dist_ji, dist_ii)  # matrix H for all h_{x_i}(x_j) values
W_x = fun_w(np.hypot(xxTX - xxRX, yyTX - yyRX))  # noise factor

#create h matrix corresponding to transmitter-receiver pair
booleReduced = np.ones(sizeK, dtype = bool)
booleReduced[indexTransPair] = False  # remove transmitter
#choose transmitter-receiver row
hVectorReduced = hMatrix[booleReduced, indexTransPair]
#repeat vector hVectorReduced as rows
hMatrixReduced = np.tile(hVectorReduced, (sizeK - 1, 1))
hMatrixReduced = hMatrixReduced.transpose()

#create reduced Palm kernels
# reduced Palm version of K matrix
KPalmReduced, _ = funPalmK(K, indexTransPair)
#calculate kernel
KReduced_h = np.sqrt(1 - hMatrixReduced.transpose()) * \
    KPalmReduced * np.sqrt(1 - hMatrixReduced)

#calculate unconditional probability for the event that transmitter's
#signal at the receiver has an SINR>tau, given the pair is active (ie
#transmitting and receiving); see equation (??) in [2]
probCovCond = np.linalg.det(np.eye(sizeK - 1) - KReduced_h) * W_x[indexTransPair]
print('probCovCondEmp = ', probCovCondEmp)
print('probCovCond = ', probCovCond)

#transmitting probabilities given by diagonals of kernel
probTX = K[indexTransPair, indexTransPair]
print('probTXEmp = ', probTXEmp)
print('probTX = ', probTX)

#calculate unconditional probability
probCov = probTX * probCovCond
print('probCovEmp = ', probCovEmp)
print('probCov = ', probCov)
###END Numerical Connection Proability START###
