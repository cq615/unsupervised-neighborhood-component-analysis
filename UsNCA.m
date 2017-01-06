function [A,Y,label]=UsNCA(X,k,labels,lamda,dd)
%input: X: input data with N samples d dimensions; k: number of clusters; labels: ground truth; 
%lamda: regularization parameter; dd: number of dimension after reduction
%ouput: A: learned transformation metric; Y: data representation after
%transformation; label: predicted labels; 
[N,d]=size(X);

no_dims = dd;
 Y = X;
A = eye(d);
%  A = zeros(d, no_dims);
for iter = 1:15  %iteration

    sumY = sum(Y .^ 2, 2);
    W = exp(bsxfun(@minus, bsxfun(@minus, 2 * (Y * Y'), sumY'), sumY));

    W(1:N+1:end) = 0;
    W = max(W, eps);

    U= Sc(W, k);
    P = bsxfun(@rdivide, W, sum(W, 2));
    P = max(P, eps);
    % PP = (P+P')/2;
    % U = maxtr(PP,k);
    % sq_sum = sqrt(sum(U.*U, 2)) + 1e-20;
    % U = U ./ repmat(sq_sum, 1, k);
    %  FF(iter) = trace(U'*P*U)-lamda .* sum(A(:) .^ 2) ./ numel(A);
    % label(:,iter) = litekmeans(U, k, 'MaxIter', 200);
    ii=0;
    while 1
        ii=ii+1;
        [label(:,iter), ~] = litekmeans(U, k, 'MaxIter', 200);
        for i = 1:k
            L(:,i)=(label(:,iter)==i);
        end
        sumL = sqrt(sum(L));
        L = bsxfun(@rdivide,L,sumL);

        %FS(iter) = trace(L'*P*L)-lamda .* sum(A(:) .^ 2) ./ numel(A);
        F(iter) = trace(L'*P*L);
        if iter ==1 || F(iter)>F(iter-1)||ii>50
            break;
        end
    end

    NMI(iter)=nmi(labels,label(:,iter));
    acc(iter) = accuracy(labels,label(:,iter));



    [EmbedDeep,mapping] = compute_mapping([label(:,iter),X], 'NCA',no_dims,lamda);

    Y = EmbedDeep;
    A = mapping.M;
end