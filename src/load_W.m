function W = load_W(data_path)

fp = fopen(data_path, 'rb');
[emb_dim] = fread(fp, 1, 'int');
[dims] = fread(fp, 1, 'int');
W = fread(fp, dims * emb_dim, 'double');
W = reshape(W,dims,emb_dim);
fclose(fp);
