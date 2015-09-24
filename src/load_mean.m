function m = load_mean(file, dim)

fp = fopen(file);

m = fread(fp, dim, 'double');

fclose(fp);
