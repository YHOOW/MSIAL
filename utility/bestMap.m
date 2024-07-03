function [newL2] = bestMap(L1, L2)
%bestmap: permute labels of L2 match L1 as good as possible
%   [newL2] = bestMap(L1,L2);

%===========    
% A(:）：将矩阵A中的每列合并成一个长的列向量
L1 = L1(:);
L2 = L2(:);
if size(L1) ~= size(L2)
    error('size(L1) must == size(L2)');
end
% unique(A):取集合A中不重复元素构成的向量
Label1 = unique(L1);
nClass1 = length(Label1);
Label2 = unique(L2);
nClass2 = length(Label2);

nClass = max(nClass1,nClass2);
G = zeros(nClass);
for i=1:nClass1
    for j=1:nClass2
        G(i,j) = length(find(L1 == Label1(i) & L2 == Label2(j)));
    end
end
% hungarian函数的功能应该是对方阵G（只能是对角矩阵，或者是上三角，下三角矩阵）进行作用，
% 返回值c就是一个1*length（G）的数组。t的值是对G对角线上的数字之和，然后加一个负号，t的值是负的
[c,t] = hungarian(-G);
newL2 = zeros(size(L2));
for i=1:nClass2
    newL2(L2 == Label2(i)) = Label1(c(i));
end


return;

%=======backup old===========

L1 = L1 - min(L1) + 1;      %   min (L1) <- 1;
L2 = L2 - min(L2) + 1;      %   min (L2) <- 1;
%===========    make bipartition graph  ============
nClass = max(max(L1), max(L2));
G = zeros(nClass);
for i=1:nClass
    for j=1:nClass
        G(i,j) = length(find(L1 == i & L2 == j));
    end
end
%===========    assign with hungarian method    ======
[c,t] = hungarian(-G);
newL2 = zeros(nClass,1);
for i=1:nClass
    newL2(L2 == i) = c(i);
end
