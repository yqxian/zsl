function [accuracy, projected_X, scores] = getProjectedX(W, X, Y, labels)

all_scores = [];
n_samples = length(labels);
%n_class = length(unique(labels));
n_class = size(Y,2);
projected_X = X * W;
projected_X = normr(projected_X);
scores = projected_X * Y;

[max_scores,predict_label] = max(scores');

%compute the confusion matrix
label_mat = sparse(labels,1:n_samples,1,n_class,n_samples);
predict_mat = sparse(predict_label,1:n_samples,1,n_class, n_samples);

conf_mat = label_mat * predict_mat';

conf_mat_diag = diag(conf_mat);
n_per_class = sum(label_mat');

%per class accuracy
accuracy = sum(conf_mat_diag ./ n_per_class') / n_class;

%per class accuracy
%accuracy = sum(conf_mat_diag) / n_samples;
