# This code compares the results from the two functions 
# funProbCovPairsDetExact and funProbCovPairsDetSamp, which respective 
# analytically calculate and statistically estimate the coverage probabilities
# in a so-called bi-pole network with determinantal scheduling.
#
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
from funProbCovPairsDetSamp import funProbCovPairsDetSamp
import numpy as np  # NumPy package for arrays, random number generation, etc
#from funSimSimpleDPP import funSimSimpleDPP
# simulate determintal point process
#from funPalmK import funPalmK  # find Palm distribution (for a single point)
#from funLtoK import funLtoK  # convert L kernel to a (normalized) K kernel

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
betaPath = 2; # path loss exponent
kappaPath = 1; # rescaling constant for path loss function
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

# helper functions
def funPathloss(r):
    return (kappaPath * (1 + r)) ** (-betaPath)  # path loss function

probCov,probTX,probCovCond = funProbCovPairsDetExact(xxTX,yyTX,\
    xxRX,yyRX,thresholdSINR,constNoise,muFading,funPathloss,L,indexTransPair)

print('probCov = ',probCov);
print('probTX = ',probTX);
print('probCovCond = ',probCovCond);

probCovEmp,probTX_Emp,probCovCondEmp = funProbCovPairsDetSamp(xxTX,yyTX,\
    xxRX,yyRX,thresholdSINR,constNoise,muFading,funPathloss,L,numbSim,indexTransPair)

print('probCovEmp = ',probCovEmp);
print('probTX_Emp = ',probTX_Emp);
print('probCovCondEmp = ',probCovCondEmp);
