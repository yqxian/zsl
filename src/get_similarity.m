%% load data
addpath('data\AWA\')
%load the mapped data if they are ready

train_X_path =  'goog_awa_train.bin';
train_label_path = 'label_train.bin';
train_Y_path = 'att_cont_train.bin';
test_X_path = 'goog_awa_test.bin';
test_label_path = 'label_test.bin';
test_Y_path = 'att_cont_test.bin';
cont_emb_filename = 'emb_goog_cont_test.bin';
%load raw data
[train_X,train_labels] = load_X(train_X_path,train_label_path);
train_Y = load_Y(train_Y_path);
[test_X, test_labels] = load_X(test_X_path,test_label_path);
test_Y = load_Y(test_Y_path);

train_idx = 1:size(train_X,1);
test_idx = train_idx(end)+1:train_idx(end)+size(test_X,1);

%load embedding matrix
W = load_W(cont_emb_filename);
%get projected X and scores matrix, accuracy
[accuracy_train, train_projected_X, train_scores] = getProjectedX(W, train_X, train_Y, train_labels);
[accuracy_test, test_projected_X, test_scores] = getProjectedX(W, test_X, test_Y, test_labels);


%% projected semantic space
figure
K_l2_project_test = gauss_kernel(test_projected_X,test_projected_X,'l2');
[test_label_sort,idx] = sort(test_labels);
K_l2_project_test = K_l2_project_test(idx,idx);
imagesc(K_l2_project_test);
title('Similarity matrix in the projected semantic space using l2 distance+gauss kernel');
print('S_project_l2_AWA', '-dpng', '-r800');

figure
K_l1_project_test = gauss_kernel(test_projected_X,test_projected_X,'l1');
[test_label_sort,idx] = sort(test_labels);
K_l1_project_test = K_l1_project_test(idx,idx);
imagesc(K_l1_project_test);
title('Similarity matrix in the projected semantic space using l1 distance+gauss kernel');
print('S_project_l1_AWA', '-dpng', '-r800');

%% prepare the scores data
% scores = [train_scores;test_scores];
figure
K_l2_score_test = gauss_kernel(test_scores,test_scores,'l2');
[test_label_sort,idx] = sort(test_labels);
K_l2_score_test = K_l2_score_test(idx,idx);
imagesc(K_l2_score_test);
title('Similarity matrix in the score space using l2 distance+gauss kernel');
print('S_score_l2_AWA', '-dpng', '-r800');

figure
K_l1_score_test = gauss_kernel(test_scores,test_scores,'l1');
[test_label_sort,idx] = sort(test_labels);
K_l1_score_test = K_l1_score_test(idx,idx);
imagesc(K_l1_score_test);
title('Similarity matrix in the score space using l1 distance+gauss kernel');
print('S_score_l1_AWA', '-dpng', '-r800');

%% plot the scoring space(normalized)
figure
K_l2_nscore_test = gauss_kernel(test_scores,test_scores,'l2');
[test_label_sort,idx] = sort(test_labels);
K_l2_nscore_test = K_l2_nscore_test(idx,idx);
imagesc(K_l2_nscore_test);
title('Similarity matrix in the normalized score space using l2 distance+gauss kernel');
print('S_nscore_l2_AWA', '-dpng', '-r800');

figure
K_l1_nscore_test = gauss_kernel(test_scores,test_scores,'l1');
[test_label_sort,idx] = sort(test_labels);
K_l1_nscore_test = K_l1_nscore_test(idx,idx);
imagesc(K_l1_nscore_test);
title('Similarity matrix in the normalized score space using l1 distance+gauss kernel');
print('S_nscore_l1_AWA', '-dpng', '-r800');
%% plot the transformed input embedding space
mapped_train_Xt = mapped_Xt(train_idx,:);
mapped_test_Xt = mapped_Xt(test_idx,:);
%plot
myplot(mapped_train_Xt,train_labels,1:27, 'Instances of Seen Classes in the Transformed Input Embedding Space');
print('InputEmbedding_AllSeen_CUB', '-dpng', '-r800');
myplot(mapped_test_Xt, test_labels,1:10, 'Instances of Unseen Classes in the Transformed Input Embedding Space');
print('InputEmbedding_AllUnseen_CUB', '-dpng', '-r800');

