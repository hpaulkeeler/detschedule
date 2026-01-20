# function KPalmReduced=funPalmK(K,indexPalm)
# This function calculates the K matrix for a Palm distribution
# conditioned on a n points existing in the statespace indexed by
# indexPalm. The method is based on the result that appears
# in the paper by Shirai and Takahashi[3]; see Theorem 6.5 and
# Corolloary 6.6 in [3].
#
# This method appears more numerically stable than using the Palm results
# for the L matrix method derived by Borodin and Rains[2].
#
# INPUTS:
# K = A square K(-matrix-)kernel, which must be (semi-)positive-definite.
#
# indexPalm = an index set for the conditioned points, where all the points
# of the underlying statespace correspond to the rows (or columns) of K.
#
# OUTPUTS:
# KPalmReduced = The reduced Palm version of the K matrix, which is a square
# matrix with dimension of size(K,1)-1.
#
# KPalm = The (non-reduced) Palm version of the K matrix, which is a square
# matrix with dimension of size(K,1).
#
# Author: H.P. Keeler, Inria/ENS, Paris, and University of Melbourne,
# Melbourne, 2019.
#
# References:
# [1] Blaszczyszyn and Keeler, "Determinantal thinning of point processes
# with network learning applications", 2018.
# [2] Borodin and Rains, "Eynard-Mehta theorem, Schur process, and their
# Pfaffian analogs", 2005
# [3] Shirai and Takahashi, "Random point fields associated with certain
# Fredholm determinants I -- fermion, poisson and boson point", 2003.

import numpy as np  # NumPy package for arrays, random number generation, etc

#####TEMP: Testing
#B=np.array([[9, 2, 1], [3, 8,2], [3, 1,7]]);
##B=np.array([[1, 2, 3,4], [5,6,7,8], [9, 10,11,12],[13, 14,15,16]]);
#
#
#L=np.matmul(B.transpose(),B);
#from funLtoK import funLtoK
#K=funLtoK(L)
#indexPalm=[0,1] #indexing starts at zero
#indexPalm=np.asarray(indexPalm); # convert to a Numpy array


def funPalmK(K, indexPalm):
    indexPalm = np.asarray(indexPalm)  # convert to a Numpy array

    #function creates reduced version of the Palm kernel
    def funPalmReducedK(K, indexPalm):

        sizeK = K.shape[0]  # number of rows/columns of K matrix

        if np.max(indexPalm) > sizeK:
            raise SystemExit('The index is too large.')

        #create Boolean array of remaining points/locations
        booleRemain = np.ones(sizeK, dtype=bool)
        booleRemain[indexPalm] = False

        if indexPalm.size == 1:
            #create Boolean array for Palm points
            boolePalm = ~booleRemain

            #create kernel for reduced Palm distribution
            KPalmReduced = K[:, booleRemain][booleRemain, :]\
                - K[:, boolePalm][booleRemain, :]\
                * K[:, booleRemain][boolePalm, :]/K[boolePalm, boolePalm]

        elif indexPalm.size > 1:
            indexPalm = np.sort(indexPalm)  # make sure index is sorted

            #call function funPalmReducedK recursively until a single point remains
            KTemp = funPalmReducedK(K, indexPalm[0])  # past the first element

            #decrease remaining indices by one
            indexPalmTemp = indexPalm[1:]-1

            KPalmReduced = funPalmReducedK(KTemp, indexPalmTemp)
        else:
            raise SystemExit('The Palm index is not a valid value.')
        return KPalmReduced

    #create non-reduced version of Palm kernel
    def funPalmNonreducedK(K, indexPalm, KPalmReduced):
        sizeK = K.shape[0]  # number of rows/columns of K matrix
        booleRemain = np.ones(sizeK, dtype=bool)
        booleRemain[indexPalm] = False
        #create indices
        indexRemain = np.arange(sizeK)[booleRemain]

        #create (non-reduced) Palm kernel
        KPalm = np.eye(sizeK)
        for i in range(KPalmReduced.shape[0]):
            KPalm[booleRemain, indexRemain[i]] = KPalmReduced[:, i]

        return KPalm

    KPalmReduced = funPalmReducedK(K, indexPalm)  # reduced Palm kernel
    # non-reduced Palm kernel
    KPalm = funPalmNonreducedK(K, indexPalm, KPalmReduced)

    return KPalmReduced, KPalm

#KPalmReduced, KPalm=funPalmK(K,indexPalm)
#print(KPalmReduced)
