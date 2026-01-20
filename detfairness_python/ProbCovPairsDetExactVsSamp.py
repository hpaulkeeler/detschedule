# Simulates a network with transmitter-receiver pairs in order to examine
# the coverage based on the signal-to-interference-plus-ratio ratio (SINR).
# The network has a random medium access control (MAC) scheme based on a
# determinantal point process, as outlined in the paper[1] by
# Blaszczyszyn, Brochard and Keeler. This code validates by simulation
# Propositions III.1 and III.2 in the paper[1]. These results give the
# probability of coverage based on the SINR value of a transmitter - receiver
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

import numpy as np

from funProbCovPairsDetExact import funProbCovPairsDetExact
from funProbCovPairsDetSamp import funProbCovPairsDetSamp

# set random seed for reproducibility
np.random.seed(1)

### START -- Parameters -- START ###
# configuration choice
choiceExample = 2  # 1 or 2 for a random (uniform) or deterministic example
numbSim = 10**4  # number of simulations
numbPairs = 5  # number of pairs
indexTransPair = 0  # index for transmitter-receiver pair (0-indexed in Python)
# the above index has to be such that 0 <= indexTransPair < numbPairs

# fading model
muFading = 1 / 3  # Rayleigh fading average
# path loss model
betaPath = 2  # pathloss exponent
kappaPath = 1  # rescaling constant for pathloss function
# SINR model
thresholdSINR = 0.1  # SINR threshold value
constNoise = 0  # noise constant

# choose kernel
choiceKernel = 1  # 1 for Gaussian (ie squared exponential); 2 for Cauchy
sigma = 0  # parameter for Gaussian and Cauchy kernel
alpha = 1  # parameter for Cauchy kernel

# simulation window parameters
# x / y dimensions
xMin = -1
xMax = 1
yMin = -1
yMax = 1
# width / height
xDelta = xMax - xMin
yDelta = yMax - yMin
### END -- Parameters -- END ###

# Simulate a random point process for the network configuration
# interferer section
if choiceExample == 1:
    # random (uniform) x / y coordinates
    # transmitters
    xxTX = xDelta * np.random.rand(numbPairs) + xMin
    yyTX = yDelta * np.random.rand(numbPairs) + yMin
    # all receivers
    xxRX = 2 * xDelta * np.random.rand(numbPairs) + xMin
    yyRX = 2 * yDelta * np.random.rand(numbPairs) + yMin
else:
    # non - random x / y coordinates
    # all transmitters
    t = 2 * np.pi * np.linspace(0, (numbPairs - 1) / numbPairs, numbPairs)
    xxTX = (1 + np.cos(5 * t + 1)) / 2
    yyTX = (1 + np.sin(3 * t + 2)) / 2
    # all receivers
    xxRX = (1 + np.sin(3 * t + 1)) / 2
    yyRX = (1 + np.cos(7 * t + 2)) / 2

# transmitter location
xxTX0 = xxTX[indexTransPair]
yyTX0 = yyTX[indexTransPair]

# receiver location
xxRX0 = xxRX[indexTransPair]
yyRX0 = yyRX[indexTransPair]

### START -- CREATE L matrix -- START ###
# all squared distances of x / y difference pairs
xx = xxTX
yy = yyTX
xxDiff = np.subtract.outer(xx, xx)
yyDiff = np.subtract.outer(yy, yy)
rrDiffSquared = xxDiff**2 + yyDiff**2

if sigma != 0:
    if choiceKernel == 1:
        # Gaussian kernel
        L = np.exp(-rrDiffSquared / sigma**2)
    elif choiceKernel == 2:
        # Cauchy kernel
        L = 1 / (1 + rrDiffSquared / sigma**2)**(alpha + 1/2)
    else:
        raise ValueError('choiceKernel has to be equal to 1 or 2.')
else:
    # use identity matrix, which is the Aloha model
    L = np.eye(numbPairs)

L = 10 * L  # scale matrix up (increases the eigenvalues ie number of points)
### END-- CREATE L matrix -- ### END

# eigen decomposition of L
eigenValL, eigenVecL = np.linalg.eig(L)

# helper functions
def funPathloss(r):
    return (kappaPath * (1 + r))**(-betaPath)

# Calculate exact coverage probability
probCov, probTX, probCovCond = funProbCovPairsDetExact(
    xxTX, yyTX, xxRX, yyRX, thresholdSINR, constNoise, muFading,
    funPathloss, L, indexTransPair
)

print("Exact results:")
print(f"  probCov     = {probCov:.6f}")
print(f"  probTX      = {probTX:.6f}")
print(f"  probCovCond = {probCovCond:.6f}")

# Calculate sampled/simulated coverage probability
probCovEmp, probTXEmp, probCovCondEmp = funProbCovPairsDetSamp(
    xxTX, yyTX, xxRX, yyRX, thresholdSINR, constNoise, muFading,
    funPathloss, L, numbSim, indexTransPair
)

print("\nSimulated results:")
print(f"  probCovEmp     = {probCovEmp:.6f}")
print(f"  probTXEmp      = {probTXEmp:.6f}")
print(f"  probCovCondEmp = {probCovCondEmp:.6f}")

print("\nDifference (Exact - Simulated):")
print(f"  probCov     = {probCov - probCovEmp:.6f}")
print(f"  probTX      = {probTX - probTXEmp:.6f}")
print(f"  probCovCond = {probCovCond - probCovCondEmp:.6f}")
