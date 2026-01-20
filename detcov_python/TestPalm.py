# This code thins a random point process using a determinantal thinning.
# Then the various Palm results are checked empirically against numerical
# results. The matrix kernels for the non-reduced and reduced Palm
# distributions are calculated by the function funPalmK.
#
# This code was originally written by H.P Keeler for the paper by
# Blaszczyszyn, Brochard and Keeler[1].
#
# If you use this code in published research, please cite paper[1].
#
# References:
#
# [1] Blaszczyszyn, Brochard and Keeler, "Coverage probability in
# wireless networks with determinantal scheduling", 2020.
#
# Author: H. Paul Keeler, 2020.

import numpy as np  # NumPy package for arrays, random number generation, etc
import matplotlib.pyplot as plt  # for plotting

from funSimSimpleDPP import funSimSimpleDPP  # simulate determinantal point process
from funPalmK import funPalmK  # find Palm distribution (for a single point)
from funLtoK import funLtoK  # convert L kernel to a (normalized) K kernel


# define function for doing determinants on scalars or matrices
def funDet(M):
    if M.size == 1:
        detM = M.flat[0]
    else:
        detM = np.linalg.det(M)
    return detM


# define function for getting submatrix
def funGetSubmatrix(M, indexRow, indexCol):
    indexRow = np.array(indexRow)
    indexCol = np.array(indexCol)
    return M[np.ix_(indexRow, indexCol)]


plt.close("all")  # close all figures

# set random seed for reproducibility
# np.random.seed(1)

### START -- Parameters -- START ###
choiceExample = 1  # 1 or 2 for a random (uniform) or deterministic example
numbSim = 10**4  # number of simulations
numbPoints = 10  # number of pairs

indexX = 0  # index for point X (0-indexed in Python, corresponds to 1 in MATLAB)
indexY = 1  # index for point Y (0-indexed in Python, corresponds to 2 in MATLAB)
indexConfig = [2, 3]  # an index for a test configuration (0-indexed)
indexConfig = np.array(indexConfig)
# above indices are bounded from above by numbPoints

# choose kernel
choiceKernel = 1  # 1 for Gaussian (ie squared exponential); 2 for Cauchy
sigma = 1  # parameter for Gaussian and Cauchy kernel
alpha = 1  # parameter for Cauchy kernel

# Simulation window parameters
xMin = -1
xMax = 1  # x dimensions
yMin = -1
yMax = 1  # y dimensions
xDelta = xMax - xMin  # rectangle width
yDelta = yMax - yMin  # rectangle height
### END -- Parameters -- END ###

# Simulate a random point process for the network configuration
if choiceExample == 1:
    # random (uniform) x/y coordinates
    xx = xDelta * np.random.rand(numbPoints) + xMin
    yy = yDelta * np.random.rand(numbPoints) + yMin
else:
    # non-random x/y coordinates
    t = 2 * np.pi * np.linspace(0, (numbPoints - 1) / numbPoints, numbPoints)
    xx = (1 + np.cos(5 * t + 1)) / 2
    yy = (1 + np.sin(3 * t + 2)) / 2

### START -- CREATE L matrix -- START ###
sizeL = numbPoints
# Calculate Gaussian or Cauchy kernel based on grid x/y values
# all squared distances of x/y difference pairs
xxDiff = np.outer(xx, np.ones(sizeL)) - np.outer(np.ones(sizeL), xx)
yyDiff = np.outer(yy, np.ones(sizeL)) - np.outer(np.ones(sizeL), yy)
rrDiffSquared = xxDiff**2 + yyDiff**2

if choiceKernel == 1:
    # Gaussian/squared exponential kernel
    L = np.exp(-rrDiffSquared / sigma**2)
elif choiceKernel == 2:
    # Cauchy kernel
    L = 1 / (1 + rrDiffSquared / sigma**2)**(alpha + 1/2)
else:
    raise Exception('choiceKernel has to be equal to 1 or 2.')
### END-- CREATE L matrix -- ### END

# Eigen decomposition
eigenValL, eigenVecL = np.linalg.eig(L)

### START Empirical Connection Probability (ie SINR>thresholdConst) START ###
# initialize boolean vectors/arrays for collecting statistics
booleX = np.zeros(numbSim, dtype=bool)  # first point exists
booleY = np.zeros(numbSim, dtype=bool)  # second point exists
booleConfig = np.zeros(numbSim, dtype=bool)  # configuration exists
numbConfig = len(indexConfig)  # number of points in configuration

# loop through all simulations
for ss in range(numbSim):
    indexDPP = funSimSimpleDPP(eigenVecL, eigenValL)

    booleX[ss] = np.any(indexDPP == indexX)
    booleY[ss] = np.any(indexDPP == indexY)

    indexTemp = np.setdiff1d(indexDPP, np.array([indexX, indexY]))  # remove X and Y

    # see if configuration exists
    booleConfig[ss] = np.sum(np.isin(indexConfig, indexTemp)) == numbConfig

booleXY = booleX & booleY
### END Empirical Connection Probability (ie SINR>thresholdConst) END ###

### START Numerical Connection Probability (ie SINR>thresholdConst) START ###
K = funLtoK(L)  # calculate K kernel from kernel L
sizeK = K.shape[0]  # number of columns/rows in kernel matrix K

# Probability of single points
probXEmp = np.mean(booleX)
probX = K[indexX, indexX]
print('probXEmp =', probXEmp)
print('probX =', probX)

probYEmp = np.mean(booleY)
probY = K[indexY, indexY]
print('probYEmp =', probYEmp)
print('probY =', probY)

probConfigEmp = np.mean(booleConfig)
probConfig = funDet(funGetSubmatrix(K, indexConfig, indexConfig))
print('probConfigEmp =', probConfigEmp)
print('probConfig =', probConfig)

# probability of configuration conditioned on a single point
probConfigXEmp = np.mean(booleConfig[booleX])
KPalmReducedX, KPalmX = funPalmK(K, indexX)
probConfigX = funDet(funGetSubmatrix(KPalmX, indexConfig, indexConfig))
print('probConfigXEmp =', probConfigXEmp)
print('probConfigX =', probConfigX)

probConfigYEmp = np.mean(booleConfig[booleY])
KPalmReducedY, KPalmY = funPalmK(K, indexY)
probConfigY = funDet(funGetSubmatrix(KPalmY, indexConfig, indexConfig))
print('probConfigYEmp =', probConfigYEmp)
print('probConfigY =', probConfigY)

# probability of configuration conditioned on two points
probConfigXYEmp = np.mean(booleConfig[booleXY]) if np.any(booleXY) else 0
KPalmReducedXY, KPalmXY = funPalmK(KPalmX, indexY)
probConfigXY = funDet(funGetSubmatrix(KPalmXY, indexConfig, indexConfig))
print('probConfigXYEmp =', probConfigXYEmp)
print('probConfigXY =', probConfigXY)

KPalmReducedYX, KPalmYX = funPalmK(KPalmY, indexX)
probConfigYX = funDet(funGetSubmatrix(KPalmYX, indexConfig, indexConfig))
print('probConfigYX =', probConfigYX)

# probability of point Y existing given X
probYGivenXEmp = np.mean(booleY[booleX]) if np.any(booleX) else 0
probYGivenX = KPalmX[indexY, indexY]
print('probYGivenXEmp =', probYGivenXEmp)
print('probYGivenX =', probYGivenX)

# probability of point X existing given Y
probXGivenYEmp = np.mean(booleX[booleY]) if np.any(booleY) else 0
probXGivenY = KPalmY[indexX, indexX]
print('probXGivenYEmp =', probXGivenYEmp)
print('probXGivenY =', probXGivenY)

# probability of configuration conditioned on point X existing and point Y
# not existing
booleXNotY = booleX & (~booleY)
probConfigXNotYEmp = np.mean(booleConfig[booleXNotY]) if np.any(booleXNotY) else 0
if probYGivenX < 1:
    probConfigXNotY = (probConfigX - probYGivenX * probConfigXY) / (1 - probYGivenX)
else:
    probConfigXNotY = 0
print('probConfigXNotYEmp =', probConfigXNotYEmp)
print('probConfigXNotY =', probConfigXNotY)
### END Numerical Connection Probability END ###

if len(indexDPP) > 0:
    ### START -- Plotting -- START ###
    plt.figure()
    plt.hold = True
    markerSizeNumb = 80  # marker size of markers
    vectorColor = np.random.rand(3)  # random vector for colors
    # Plot point process
    plt.plot(xx, yy, 'ko', markersize=markerSizeNumb/6, fillstyle='none')
    # Plot determinantally-thinned point process
    plt.plot(xx[indexDPP], yy[indexDPP], '.', markersize=markerSizeNumb/3, color=vectorColor)
    plt.grid(True)
    plt.axis('square')
    plt.xticks([])
    plt.yticks([])
    plt.legend(['Original point process', 'Determinantal subset'])
    plt.show()
    ### END -- Plotting -- END ###
