function Y = load_Y(data_path)

fp = fopen(data_path);
[ndim] = fread(fp, 1, 'int');
[nclass] = fread(fp, 1, 'int');
Y = fread(fp, ndim * nclass, 'double');
Y = reshape(Y,ndim,nclass);
fclose(fp);