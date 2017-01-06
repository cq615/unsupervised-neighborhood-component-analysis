function score = accuracy(true_labels, cluster_labels)
%ACCURACY Compute clustering accuracy using the true and cluster labels and
%   return the value in 'score'.
%
%   Input  : true_labels    : N-by-1 vector containing true labels
%            cluster_labels : N-by-1 vector containing cluster labels
%
%   Output : score          : clustering accuracy

% Compute the confusion matrix 'cmat', where
%   col index is for true label (CAT),
%   row index is for cluster label (CLS).
n = length(true_labels);


% Calculate accuracy
nC=length(unique(true_labels));
if nC==2
    tmp=100*sum(true_labels==cluster_labels)/n;
    score=max(tmp,100-tmp);
else
    cat = spconvert([(1:n)' true_labels ones(n,1)]);
    cls = spconvert([(1:n)' cluster_labels ones(n,1)]);
    cls = cls';
    cmat = full(cls * cat);
    [match, cost] = hungarian(-cmat);
    score = (-cost/n);
end
