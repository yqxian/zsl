function W = construct_knn(Ix,k)

n = size(Ix,1);
kIx = Ix(:,2:k+1)';
jx = kIx(:);
ix = repmat([1:n]',1,k)';
ix = ix(:);
W = sparse(ix,jx,1);
W = 1/2*(W+W');
