function F = maxtr(P,k)
%   P = max(P, eps);
[U,D]=eigs(P,k);
D = real(D);
[dummy,order] = sort(diag(-D));
U = U(:,order);
F = U(:,1:k);