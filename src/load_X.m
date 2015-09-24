function [X, label] = load_X(data_path1, data_path2)

fp = fopen(data_path1);
nsample = fread(fp, 1, 'int');
ndim = fread(fp, 1, 'int');
X = fread(fp, nsample * ndim, 'double');
fclose(fp);

X = reshape(X, [ndim,nsample]);
X = X';

fp = fopen(data_path2);
label = fread(fp, nsample, 'double');
fclose(fp);