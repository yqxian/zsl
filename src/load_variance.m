function m = load_variance(file, dim)

fp = fopen(file);

m = fread(fp, dim, 'double');

fclose(fp);
