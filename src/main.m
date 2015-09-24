dataset = {'Dogs113'};
wordembedding = {'word2vec','glove','wordnet','bow'};
dataroot = '/BS/Deep_Fragments/work/MSc/data/';

accuracy_mani = zeros(length(dataset),length(wordembedding));

knn = 50;
sigma = [0.2 0.4 0.6 1.0 1.3 1.5 2 2.5];
alpha = [0.4 0.5 0.6 0.7 0.8 0.9];
delta = 0.15;
dist_type = 'l1';

for i=1:length(dataset)
	disp(['Start to evaluate on...' dataset{i}]);
	disp('Loading data...');

	val_X_path = strcat(dataroot, dataset{i}, '/goog_val.bin');
	val_label_path = strcat(dataroot, dataset{i}, '/label_val.bin');
	
	test_X_path = strcat(dataroot, dataset{i}, '/goog_test.bin');
	test_label_path = strcat(dataroot, dataset{i}, '/label_test.bin');
	
	xval_max_path = strcat(dataroot, dataset{i}, '/xtrain_max');
	xtest_max_path = strcat(dataroot, dataset{i}, '/xtrainval_max');

	xval_mean_path = strcat(dataroot, dataset{i}, '/xtrain_mean');
	xtest_mean_path = strcat(dataroot, dataset{i}, '/xtrainval_mean');

	xval_variance_path = strcat(dataroot, dataset{i}, '/xtrain_variance');
	xtest_variance_path = strcat(dataroot, dataset{i}, '/xtrainval_variance'); 

	%load raw data
	[val_X,val_labels] = load_X(val_X_path,val_label_path);
 
	[test_X, test_labels] = load_X(test_X_path,test_label_path);
	
	dim = size(val_X, 2);
	val_nsample = size(val_X, 1);
	test_nsample = size(test_X, 1);

	xval_max = load_max(xval_max_path);
	xtest_max	= load_max(xtest_max_path);
	
	xval_mean = load_mean(xval_mean_path, dim);
	xval_mean = xval_mean';
	xtest_mean = load_mean(xtest_mean_path, dim);
	xtest_mean = xtest_mean';
	xval_variance = load_variance(xval_variance_path, dim);
	xval_variance = xval_variance';
	xval_variance(1) = 1;

	xtest_variance = load_variance(xtest_variance_path, dim);
	xtest_variance = xtest_variance';
	xtest_variance(1) = 1;
	% zero score
	val_X = (val_X - repmat(xval_mean,[val_nsample 1])) ./ repmat(xval_variance,[val_nsample 1]);	
	val_X = val_X / xval_max;

	test_X = (test_X - repmat(xtest_mean,[test_nsample 1])) ./ repmat(xtest_variance,[test_nsample 1]);	
	test_X = test_X / xtest_max;

	for e=1:length(wordembedding)
		%load embedding matrix
		val_Y_path = strcat(dataroot, dataset{i}, '/att_',wordembedding{e},'_val.bin');
		val_Y = load_Y(val_Y_path);
		test_Y_path = strcat(dataroot, dataset{i}, '/att_',wordembedding{e},'_test.bin');
		test_Y = load_Y(test_Y_path);
		
		emb_test_filename = strcat(dataroot, dataset{i}, '/emb_mat_',wordembedding{e},'_test');
 		emb_val_filename = strcat(dataroot, dataset{i}, '/emb_mat_',wordembedding{e},'_val');
		disp(emb_val_filename);
		W_val = load_W(emb_val_filename);
		W_test = load_W(emb_test_filename);

		disp('Compute projected X...');
		%get projected X and scores matrix, accuracy
		[accuracy_val, val_projected_X, val_scores] = getProjectedX(W_val, val_X, val_Y, val_labels);
		[accuracy_test, test_projected_X, test_scores] = getProjectedX(W_test, test_X, test_Y, test_labels);

		disp('Grid search for sigma and alpha on the validation set...');
		grid_accuracy = zeros(length(sigma),length(alpha));
		n_samples = size(val_X, 1);
		n_class = length(unique(val_labels));

		for j=1:length(sigma)
   
  		disp(['sigma= ' num2str(sigma(j)) ', compute the gaussian kernel...']);
    	W_full = gauss_kernel(val_projected_X,val_projected_X,dist_type, sigma(j));
    	%diagonal elements should be 0
			W_full(1:n_samples+1:n_samples*n_samples) = 0;

    	disp('Construct knn graph...');
    	[sorted_W,idx_W] = sort(W_full, 'descend');
    	W_knn = construct_knn(idx_W, knn);
    	W_knn = W_knn .* W_full; 

    	Y = val_scores;
    	Y(Y<delta) = 0;
    	disp('');
    
    	for k=1:length(alpha)
        disp(['alpha= ' num2str(alpha(k)) ', Label propogation...']);
        predict_label = ssl(W_knn, Y, alpha(k));
        
        disp('Evaluation...');
        label_mat = sparse(val_labels',1:n_samples,1,n_class,n_samples);
        predict_mat = sparse(predict_label,1:n_samples,1,n_class, n_samples);

        conf_mat = label_mat * predict_mat';

        conf_mat_diag = diag(conf_mat);
        n_per_class = sum(label_mat');

        %per class accuracy
        grid_accuracy(j,k) = sum(conf_mat_diag ./ n_per_class') / n_class;
    	end
		end

		[sigma_idx, alpha_idx] = find(grid_accuracy==max(max(grid_accuracy)));
		best_alpha = alpha(alpha_idx);
		best_sigma = sigma(sigma_idx);

		disp(['Selected alpha= ' num2str(alpha(alpha_idx)) ', sigma= ' num2str(sigma(sigma_idx))]);

	%% Start to evaluate on the test set
		n_samples = size(test_X, 1);
		n_class = length(unique(test_labels));

		disp(['sigma= ' num2str(best_sigma) ', compute the gaussian kernel...']);
 		W_full = gauss_kernel(test_projected_X,test_projected_X,dist_type, best_sigma);
 		%diagonal elements should be 0
 		W_full(1:n_samples+1:n_samples*n_samples) = 0;
 
 		disp('Construct knn graph...');
 		[sorted_W,idx_W] = sort(W_full, 'descend');
 		W_knn = construct_knn(idx_W, knn);
 		W_knn = W_knn .* W_full;
 
 		Y = test_scores;
		Y(Y<delta) = 0;
		disp('');

		disp(['alpha= ' num2str(best_alpha) ', Label propogation...']);
  	predict_label = ssl(W_knn, Y, best_alpha);

  	disp('Evaluation...');
		label_mat = sparse(test_labels',1:n_samples,1,n_class,n_samples);
  	predict_mat = sparse(predict_label,1:n_samples,1,n_class, n_samples);

  	conf_mat = label_mat * predict_mat';

  	conf_mat_diag = diag(conf_mat);
  	n_per_class = sum(label_mat');

  	%per class accuracy
  	accuracy_mani(i,e) = sum(conf_mat_diag ./ n_per_class') / n_class;
	
		display(['Accurayc for SJE using ' wordembedding{e} ' is ' num2str(accuracy_test)]);

		display(['Accurayc for manifold learning is ',num2str(accuracy_mani(i,e))]);

		save(strcat(dataset{i}, wordembedding{e}));
	end
end
