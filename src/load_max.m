function m = load_max(file)

fp = fopen(file);

m = fread(fp, 1, 'double');

fclose(fp);
