function K = gauss_kernel(X, Y, dist_type, sigma)

if(strcmp(dist_type, 'l2'))
    D = dist_euclidean(X,Y);    
elseif(strcmp(dist_type, 'l1'))
    D = dist_l1(X,Y);
elseif(strcmp(dist_type, 'cos'))
    D = dist_cos(X,Y);
else
    display('No such distance!');
end

%gamma =  median(D(:));                       %median of all squared distances
%gamma = 1;
lambda = 1 / (2*(sigma^2));
K = exp(-lambda*D);
