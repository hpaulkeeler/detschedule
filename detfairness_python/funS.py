# S = funS(xx,yy,choiceKernel,paramKernel)
#
# This code generates a similarity matrix S based on Cartesian
# coordinates xx and yy.
#
# INPUTS:
#
# xx and yy are arrays for the Cartesian coordinates of the points.
#
# choiceKernel is a number between 1 and 3 for choosing which radial
# function for the similarity matrix. 1 is Gaussian, 2 is Cauchy, and
# 3 is Bessel.
#
# paramKernel is an array of parameters for the radial functions. For the
# aforementioned three radial functions, the first value is the scale
# parameter sigma, while the second value is is a second paramter for the
# Cauchy function. If the paramKernel = 0, then the identity matrix is returned
# for the similarity matrix S.
#
# OUTPUTS:
#
# S is the similarity matrix.
#
# This code was originally written by H.P. Keeler for the paper[1] by
# Blaszczyszyn and Keeler, which studies determinantal scheduling in wireless
# networks.
#
# If you use this code in published research, please cite paper[1].
#
# More details are given in paper[1] and paper[2].
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
import scipy.special as sp; # for Bessel function jv
import  scipy.spatial as spatial

def funS(xx,yy,choiceKernel,paramKernel):
    # xx / yy need to be column vectors
    xx = np.ravel(xx);
    yy = np.ravel(yy);

    # retrieve kernel parameters
    sigma = paramKernel[0];
    alpha = paramKernel[-1]; #if there is a second parameter

    distBetween = spatial.distance.pdist(np.vstack((xx,yy)).T);# inter - point distances
    distBetweenMean = np.mean(distBetween);# average inter - point distance
    sigma = sigma * distBetweenMean; # rescale sigma

    sizeS = xx.size; # number of columns / rows

    ### NOTE:
    # As sigma approaches zero, S approaches the identity matrix
    # As sigma approaches infinity, S approaches a matrix of ones, which has a
    # zero determinant (meaning its ill - conditioned in terms of inverses)

    ### START - Create similarity matrix S - START###
    if sigma!=0:
        # all squared distances of x / y difference pairs
        xxDiff = np.outer(xx, np.ones((sizeS,))) - np.outer( np.ones((sizeS,)),xx);
        yyDiff = np.outer(yy, np.ones((sizeS,))) - np.outer( np.ones((sizeS,)),yy);
        rrDiffSquared=(xxDiff**2 + yyDiff**2);
        if choiceKernel==1:
            ## Gaussian kernel
            # See the paper by Lavancier, Moller and Rubak (2015)
            S = np.exp(-(rrDiffSquared) / sigma**2);
        elif choiceKernel==2:
            ## Cauchy kernel
            # See the paper by Lavancier, Moller and Rubak (2015)
            #alpha = 1; # an additional parameter for the Cauchy (ie second) kernel
            S = 1 / (1 + rrDiffSquared / sigma**2)**(alpha + 1/2);
        elif choiceKernel==3:
            ## Bessel kernel
            # See the Supplementary Material for the paper by  Biscio and
            # Lavancier (2016), page 2007. Kernel CI, where sigma has been
            # introduced as a scale parameter, similar to the Gaussian and
            # Cauchy cases.
            rrDiff = np.sqrt(rrDiffSquared);
            # Bessel (simplified) kernel
            np.fill_diagonal(rrDiff,1); # prevent zero division
            S = sp.jv(1,2 * np.sqrt(np.pi) * rrDiff / sigma) / (np.sqrt(np.pi) * rrDiff / sigma);
            # need to rescale to ensure that diagonal entries are ones.
            np.fill_diagonal(S,1); # set to correct value
    else:
        # use identity matrix, which is the Aloha model
        S = np.eye(sizeS);
    ### END - Create similarity matrix S - END###
    return S
