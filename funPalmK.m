% function KPalmReduced=funPalmK(K,indexPalm)
% This function calculates the K matrix for a Palm distribution
% conditioned on points existing in the statespace indexed by
% indexPalm. The method is based on the result that appears
% in the paper by Shirai and Takahashi[3]; see Theorem 6.5 and
% Corolloary 6.6 in [3].
%
% This method appears more numerically stable than using the Palm results
% for the L matrix method derived by Borodin and Rains[2].
% 
% The code was originally written for the paper by Blaszczyszyn and
% Keeler[1], and then later modified for the paper by Blaszczyszyn, 
% Brochard and Keeler[4] so it could do multiple points.
%
% INPUTS:
% K = The kernel matrix of a determinantal point process. 
%
% indexPalm = an index set for the conditioned point, where all the points
% of the underlying statespace correspond to the rows (or columns) of K.
%
% OUTPUTS:
% KPalmReduced= The (reduced) Palm version of the K matrix, which is a 
% square matrix with dimension of size(K,1)-1.
%
% KPalm= The (non-reduced) Palm version of the K matrix, which is a square
% matrix with dimension of size(K,1).
%
% Author: H. Paul Keeler, 2020.
%
% References:
% [1] Blaszczyszyn and Keeler, "Determinantal thinning of point processes
% with network learning applications", 2018.
% [2] Borodin and Rains, "Eynard-Mehta theorem, Schur process, and their
% Pfaffian analogs", 2005
% [3] Shirai and Takahashi, "Random point fields associated with certain
% Fredholm determinants I -- fermion, poisson and boson point", 2003.
% [4] B\laszczyszyn, Brochard and Keeler, "Coverage probability in
% wireless networks with determinantal scheduling", 2020.
%
% %%TEMP: Testing
% clearvars; close all; clc
% B=[9, 2, 1; 3, 8,2; 3, 1,7]
% L=B'*B;
% K=funLtoK(L)
%
% %K=[1, 2, 3,4; 5,6,7,8; 9, 10,11,12;13, 14,15,16];
%
% indexPalm=[1,2];


function [KPalmReduced,KPalm]=funPalmK(K,indexPalm)

indexPalm=indexPalm(:);

KPalmReduced=funPalmReducedK(K,indexPalm); %reduced Palm kernel
KPalm=funPalmNonreducedK(K,indexPalm,KPalmReduced); %non-reduced Palm kernel

%create reduced version of Palm kernel
    function KPalmReduced=funPalmReducedK(K,indexPalm)
        
        sizeK=size(K,1); %number of rows/columns of K matrix
        
        if max(indexPalm)>sizeK
            error('The index is too large.');
        end
        
        %create Boolean array of remaining points/locations
        booleRemain=true(1,sizeK);
        booleRemain(indexPalm)=false;
        
        if length(indexPalm)==1
            %create Boolean array for Palm points
            boolePalm=~booleRemain;
            
            %create kernel for a reduced Palm distribution (one point)
            KPalmReduced=K(booleRemain,booleRemain)...
                -K(booleRemain,boolePalm).*K(boolePalm,booleRemain)/K(boolePalm,boolePalm);
            
        elseif length(indexPalm)>1
            indexPalm=sort(indexPalm); %make sure index is sorted
            
            %call function recurisively until a single point remains
            KTemp=funPalmReducedK(K,indexPalm(1)); %past the first element
            
            %decrease remaining indices by one
            indexPalmTemp=indexPalm(2:end)-1;
            KPalmReduced=funPalmReducedK(KTemp,indexPalmTemp);
        else
            error('The index is not a valid value.');
        end
    end

%create non-reduced version of Palm kernel
    function KPalm= funPalmNonreducedK(K,indexPalm,KPalmReduced)
        sizeK=size(K,1); %number of rows/columns of K matrix
        
        %create Boolean array of remaining points/locations
        booleRemain=true(1,sizeK);
        booleRemain(indexPalm)=false;
        
        %create (non-reduced) Palm kernel
        KPalm=eye(sizeK);
        KPalm(booleRemain,booleRemain)=KPalmReduced;
    end

end
