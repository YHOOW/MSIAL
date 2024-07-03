function [X, sigma] = project_simplex(B)

% Project onto the probability simplex
% min_X ||X-B||_F
% s.t Xe=e, X>=0 where e is the constant one vector.
%
% ---------------------------------------------
% Input:
%       B       -    n*d matrix
%
% Output:
%       X       -    n*d matrix
% 

[n,m] = size(B);
A = repmat(1:m,n,1);
B_sort = sort(B,2,'descend');
 % B = cumsum(A):这种用法返回数组不同维数的累加和,B = cumsum(A,dim),这种调用格式返回A中由标量dim所指定的维数的累加和
cum_B = cumsum(B_sort,2);
sigma = B_sort-(cum_B-1)./A;
tmp = sigma>0;
idx = sum(tmp,2);
tmp = B_sort-sigma;
sigma = diag(tmp(:,idx));
sigma = repmat(sigma,1,m);
X = max(B-sigma,0);

