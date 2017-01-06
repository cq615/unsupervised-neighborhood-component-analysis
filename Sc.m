function U = Sc(W,t)
% clustering
N = size(W,1);
for i=1:N
     D(i,i)=sum(W(i,:));
end
L=D-W;
L=D\L;
[u, lamda]=eigs(L,t,'SM');

y=diag(lamda);
[~, d]=sort(y);


dim=d(1:t);
U=u(:,dim);
sq_sum = sqrt(sum(U.*U, 2)) + 1e-20;
U = U ./ repmat(sq_sum, 1, t);
