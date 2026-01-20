% K=funLtoK(L)
% The function funLtoK(L) converts a kernel L matrix into  a (normalized)
% kernel K matrix. The K matrix has to be semi-positive definite.

% clearvars; clc%
% B=[3, 2, 1; 4, 5,6; 9, 8,7];
% L=B'*B;
% K=funLtoK(L)
%   K =
%
%     0.7602    0.3348   -0.0906
%     0.3348    0.3320    0.3293
%    -0.0906    0.3293    0.7492

function K=funLtoK(L)
%METHOD 1 -- using eigen decomposition.
%This method doesn't need inverse calculating and seems to more stable.
%tic;
[eigenVectLK,eigenValL]=eig(L); %eigen decomposition
eigenValL=(diag(eigenValL)); %eigenvalues of L as vector
eigenValK = eigenValL./(1+eigenValL); %eigenvalues of K
eigenValK=diag(eigenValK); %%eigenvalues of L as diagonal matrix
K=eigenVectLK*eigenValK*(eigenVectLK'); %recombine from eigen components
K=real(K); %make sure all values are real
%toc;

end

% %METHOD 2 -- standard approach.
% %Slightly faster, seems to be less stable.
% %tic;
% K=L/(eye(size(L,1))+L); %
% %toc;

% %METHOD 3 -- LDL decomposition and Woodbury/Sherman-Morrison formula.
% %LDL decomposition
% [leftL, diagL]=ldl(L);
% diagInverseL=diag(1./diag(diagL));
% rightL=leftL';
% %Use the Woodbury/Sherman-Morrison formula
% K=L*(eye(size(L,1))-leftL/(diagInverseL+rightL*leftL)*(rightL));
%
% %funSherman=@(A,U,C,V)inv(A)-inv(A)*U/(inv(C)+V*inv(A)*U)*V*inv(A);

