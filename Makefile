# !/usr/bin/make
# Makefile
#
# Zeynep Akata
# e-mail: akata@mpi-inf.mpg.de
# 
# Max Planck Institute for Informatics
# Campus E1 4
# 66123 Saarbrucken
# GERMANY


# Path of the directory that you downloaded the code and the data
PATH_ALL=/BS/Deep_Fragments/work/MSc
BIN=$(PATH_ALL)/msje_latefusion/bin
TMP=$(PATH_ALL)/msje_latefusion/results_$(DATASET)
DATA=$(PATH_ALL)/data/$(DATASET)
# INT = wherever the learned embeddings are going to be saved
INT=$(DATA)/MSJE_LATEFUSION

EMB1=cont
EMB2=word2vec
EMB3=glove
EMB4=wordnet
EMB5=bow
#########################################################################
#			SJE ZERO_SHOT 					#
#########################################################################
CUB:
	make run_dataset DATASET=CUB 

AWA:
	make run_dataset DATASET=AWA

Dogs:
	make run_dataset DATASET=Dogs113

run_dataset:
	-rm -r $(TMP)
	mkdir -p $(TMP)
ifeq ($(DATASET), Dogs113)
		make run_allcombine2
else
		make run_allcombine
endif

run_allcombine2:
	screen -S msje_latefusion_${DATASET}_$(EMB5)$(EMB2) -m -d make run_batch ATT1=$(EMB5) ATT2=$(EMB2)
	screen -S msje_latefusion_${DATASET}_$(EMB5)$(EMB3) -m -d make run_batch ATT1=$(EMB5) ATT2=$(EMB3)
	screen -S msje_latefusion_${DATASET}_$(EMB5)$(EMB4) -m -d make run_batch ATT1=$(EMB5) ATT2=$(EMB4)
	screen -S msje_latefusion_${DATASET}_$(EMB2)$(EMB3) -m -d make run_batch ATT1=$(EMB2) ATT2=$(EMB3)
	screen -S msje_latefusion_${DATASET}_$(EMB2)$(EMB4) -m -d make run_batch ATT1=$(EMB2) ATT2=$(EMB4)
	screen -S msje_latefusion_${DATASET}_$(EMB3)$(EMB4) -m -d make run_batch ATT1=$(EMB3) ATT2=$(EMB4)

run_allcombine:
	screen -S msje_latefusion_${DATASET}_$(EMB1)$(EMB2) -m -d make run_batch ATT1=$(EMB1) ATT2=$(EMB2)
	screen -S msje_latefusion_${DATASET}_$(EMB1)$(EMB3) -m -d make run_batch ATT1=$(EMB1) ATT2=$(EMB3)
	screen -S msje_latefusion_${DATASET}_$(EMB1)$(EMB4) -m -d make run_batch ATT1=$(EMB1) ATT2=$(EMB4)
	screen -S msje_latefusion_${DATASET}_$(EMB2)$(EMB3) -m -d make run_batch ATT1=$(EMB2) ATT2=$(EMB3)
	screen -S msje_latefusion_${DATASET}_$(EMB2)$(EMB4) -m -d make run_batch ATT1=$(EMB2) ATT2=$(EMB4)
	screen -S msje_latefusion_${DATASET}_$(EMB3)$(EMB4) -m -d make run_batch ATT1=$(EMB3) ATT2=$(EMB4)

run_batch:
	make run_val FEAT=goog
	make run_test FEAT=goog
	
run_val: 
	make $(TMP)/msje_latefusion_${ATT1}${ATT2}_val.txt

$(TMP)/msje_latefusion_${ATT1}${ATT2}_val.txt:
	$(BIN)/msje_latefusion -val\
		$(DATA)/train_${FEAT}_zsh.train.bin.gz $(DATA)/val_${FEAT}_zsh.test.bin.gz \
		$(DATA)/att_${ATT1}_train.bin $(DATA)/att_${ATT1}_val.bin \
		$(DATA)/att_${ATT2}_train.bin $(DATA)/att_${ATT2}_val.bin \
		$(INT)/emb_mat_${ATT1}_val $(INT)/emb_mat_${ATT2}_val $(INT)/emb_tensor_${ATT1}${ATT2}_val $(DATA)/xtrain_mean $(DATA)/xtrain_variance $(DATA)/xtrain_max \
	>> $@

run_one_test:
	make $(TMP)/msje_latefusion_${ATT1}${ATT2}_test.txt

$(TMP)/msje_latefusion_${ATT1}${ATT2}_test.txt:
	$(BIN)/msje_latefusion -alpha ${ALPHA} -beta ${BETA}\
		$(DATA)/trainval_${FEAT}_zsh.train.bin.gz $(DATA)/test_${FEAT}_zsh.test.bin.gz \
		$(DATA)/att_${ATT1}_trainval.bin $(DATA)/att_${ATT1}_test.bin \
		$(DATA)/att_${ATT2}_trainval.bin $(DATA)/att_${ATT2}_test.bin \
		$(INT)/emb_mat_${ATT1}_test $(INT)/emb_mat_${ATT2}_test $(INT)/emb_tensor_${ATT1}${ATT2}_test $(DATA)/xtrainval_mean $(DATA)/xtrainval_variance $(DATA)/xtrainval_max \
	>> $@

#########################################################################
#				SCORING	    			        #
#########################################################################

run_test:
	-rm $(TMP)/param_${ATT1}${ATT2}.txt
	less $(TMP)/msje_latefusion_${ATT1}${ATT2}_val.txt | tail -1 > $(TMP)/param_${ATT1}${ATT2}.txt
	make run_one_test ALPHA=`awk '{print $$2}' $(TMP)/param_${ATT1}${ATT2}.txt` BETA=`awk '{print $$3}' $(TMP)/param_${ATT1}${ATT2}.txt`
