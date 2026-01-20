# K = funLtoK(L)
#
# The function funLtoK(L) converts a kernel L matrix into a (normalized)
# kernel K matrix. The L matrix has to be semi - positive definite.
#
# INPUTS: L is a semi - positive matrix L with positive eigenvalues.
#
# OUTPUT: L is a semi - positive matrix L with eigenvalues on the unit interval.
#
# EXAMPLE:
# B = np.array([[3, 2, 1], [4, 5,6], [9, 8,7]]);
# L = B'*B;
# K = funLtoK(L)
#   K =
#
#     0.7602    0.3348   -0.0906
#     0.3348    0.3320    0.3293
#    -0.0906    0.3293    0.7492
#
# This code was used by H.P. Keeler for the paper[1] by 
# Blaszczyszyn and Keeler, which studies determinantal scheduling in wireless 
# networks.  
# 
# It was origianlly written by H.P. Keeler for the paper[2] by Blaszczyszyn, 
# Brochard and Keeler.
#
# If you use this code in published research, please cite paper[1] or [2].
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

import numpy as np # NumPy package for arrays, random number generation, etc

def funLtoK(L):
    # METHOD 1 -- using eigen decomposition.
    # This method doesn't need inverse calculating and seems to more stable.    
    eigenValL,eigenVectLK = np.linalg.eig(L); # eigen decomposition    
    eigenValK = eigenValL / (1 + eigenValL); # eigenvalues of K
    eigenValK = np.diagflat(eigenValK); ## eigenvalues of L as diagonal matrix
    K = np.matmul(np.matmul(eigenVectLK,eigenValK),eigenVectLK.transpose()); # recombine from eigen components    
    K = np.real(K); # make sure all values are real        
    return K