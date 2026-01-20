# K=funLtoK(L)
# The function funLtoK(L) converts a kernel L matrix into  a (normalized)
# kernel K matrix. The K matrix has to be semi-positive definite.

import numpy as np  # NumPy package for arrays, random number generation, etc

##TEMP: Testing
#B=np.array([[3, 2, 1], [4, 5,6], [9, 8,7]]);
#
#L=np.matmul(B.transpose(),B);
#
#


def funLtoK(L):
    #METHOD 1 -- using eigen decomposition.
    #This method doesn't need inverse calculating and seems to more stable.
    eigenValL, eigenVectLK = np.linalg.eig(L)  # eigen decomposition
    eigenValK = eigenValL/(1+eigenValL)  # eigenvalues of K
    eigenValK = np.diagflat(eigenValK)  # eigenvalues of L as diagonal matrix
    # recombine from eigen components
    K = np.matmul(np.matmul(eigenVectLK, eigenValK), eigenVectLK.transpose())
    K = np.real(K)  # make sure all values are real
    return K


#K=funLtoK(L)
# K=
#array([[ 0.76022099,  0.33480663, -0.09060773],
#       [ 0.33480663,  0.3320442 ,  0.32928177],
#       [-0.09060773,  0.32928177,  0.74917127]])

# #METHOD 2 -- standard approach.
# #Slightly faster, seems to be less stable.
# K=np.matmul(L,np.linalg.inv(L+np.eye(L.shape[0])));
