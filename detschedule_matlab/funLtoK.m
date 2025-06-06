% K = funLtoK(L)
%
% The function funLtoK(L) converts a kernel L matrix into a (normalized)
% kernel K matrix. The L matrix has to be semi - positive definite.
%
% INPUTS: L is a semi - positive matrix L with positive eigenvalues.
%
% OUTPUT: L is a semi - positive matrix L with eigenvalues on the unit interval.
%
% EXAMPLE:
% B = [3, 2, 1; 4, 5,6; 9, 8,7];
% L = B'*B;
% K = funLtoK(L)
%   K =
%
%     0.7602    0.3348   -0.0906
%     0.3348    0.3320    0.3293
%    -0.0906    0.3293    0.7492
%
% This code was used by H.P. Keeler for the paper[1] by 
% Blaszczyszyn and Keeler, which studies determinantal scheduling in wireless 
% networks.  
% 
% It was origianlly written by H.P. Keeler for the paper[2] by Blaszczyszyn, 
% Brochard and Keeler.
%
% If you use this code in published research, please cite paper[1] or [2].
%
% References:
%
% [1] Blaszczyszyn and Keeler, "Adaptive determinantal scheduling with
% fairness in wireless networks", 2025.
%
% [2] Blaszczyszyn, Brochard and Keeler, "Coverage probability in
% wireless networks with determinantal scheduling", 2020.
%
% Author: H. Paul Keeler, 2025.

function K = funLtoK(L)
% METHOD 1 -- using eigendecomposition.
% This method doesn't need to calcualte the inverse and seems to be more stable.
[eigenVectLK,eigenValL]=eig(L); % eigen decomposition
eigenValL=(diag(eigenValL)); % eigenvalues of L as vector
eigenValK = eigenValL./(1 + eigenValL); % eigenvalues of K
eigenValK = diag(eigenValK); %% eigenvalues of L as diagonal matrix
K = eigenVectLK * eigenValK * (eigenVectLK'); % recombine from eigen components
K = real(K); % make sure all values are real

end

