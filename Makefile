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
BIN=$(PATH_ALL)/msje_lowrank/bin
TMP=$(PATH_ALL)/msje_lowrank/results_$(DATASET)
DATA=$(PATH_ALL)/data/$(DATASET)
# INT = wherever the learned embeddings are going to be saved
FINAL=$(DATA)/MSJE_LOWRANK/TENSOR_$(ETA)_$(NITER)_$(RANK)_$(ATT1)$(ATT2)

EPOCHS=30

EMB1=cont
EMB2=word2vec
EMB3=glove
EMB4=wordnet
EMB5=bow
#########################################################################
#			SJE ZERO_SHOT 					#
#########################################################################
CUB:
	-rm -rf $(PATH_ALL)/data/CUB/MSJE_LOWRANK
	make run_dataset DATASET=CUB 

AWA:
	-rm -rf $(PATH_ALL)/data/AWA/MSJE_LOWRANK
	make run_dataset DATASET=AWA

Dogs:
	-rm -rf $(PATH_ALL)/data/Dogs113/MSJE_LOWRANK
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
	screen -S msje_lowrank_${DATASET}_$(EMB5)$(EMB2) -m -d make run_batch ATT1=$(EMB5) ATT2=$(EMB2)
	screen -S msje_lowrank_${DATASET}_$(EMB5)$(EMB3) -m -d make run_batch ATT1=$(EMB5) ATT2=$(EMB3)
	screen -S msje_lowrank_${DATASET}_$(EMB5)$(EMB4) -m -d make run_batch ATT1=$(EMB5) ATT2=$(EMB4)
	screen -S msje_lowrank_${DATASET}_$(EMB2)$(EMB3) -m -d make run_batch ATT1=$(EMB2) ATT2=$(EMB3)
	screen -S msje_lowrank_${DATASET}_$(EMB2)$(EMB4) -m -d make run_batch ATT1=$(EMB2) ATT2=$(EMB4)
	screen -S msje_lowrank_${DATASET}_$(EMB3)$(EMB4) -m -d make run_batch ATT1=$(EMB3) ATT2=$(EMB4)

run_allcombine:
	screen -S msje_lowrank_${DATASET}_$(EMB1)$(EMB2) -m -d make run_batch ATT1=$(EMB1) ATT2=$(EMB2)
	screen -S msje_lowrank_${DATASET}_$(EMB1)$(EMB3) -m -d make run_batch ATT1=$(EMB1) ATT2=$(EMB3)
	screen -S msje_lowrank_${DATASET}_$(EMB1)$(EMB4) -m -d make run_batch ATT1=$(EMB1) ATT2=$(EMB4)
	screen -S msje_lowrank_${DATASET}_$(EMB2)$(EMB3) -m -d make run_batch ATT1=$(EMB2) ATT2=$(EMB3)
	screen -S msje_lowrank_${DATASET}_$(EMB2)$(EMB4) -m -d make run_batch ATT1=$(EMB2) ATT2=$(EMB4)
	screen -S msje_lowrank_${DATASET}_$(EMB3)$(EMB4) -m -d make run_batch ATT1=$(EMB3) ATT2=$(EMB4)

run_batch:
	for rank in 100 200 400 600 800 1000 1500 2000; do \
		make run_one_val NITER=$(EPOCHS) LBD=1e-5 FEAT=goog ETA=1e-3 RANK=$${rank};\
	done
	make run_test LBD=1e-5 FEAT=goog NITER=$(EPOCHS) ETA=1e-3
	
run_one_val: 
	make $(TMP)/msje_lowrank_${RANK}_${ETA}_${NITER}_${FEAT}_${ATT1}${ATT2}_val.txt

$(TMP)/msje_lowrank_${RANK}_${ETA}_${NITER}_${FEAT}_${ATT1}${ATT2}_val.txt:
	mkdir -p $(FINAL)
	$(BIN)/msje_lowrank -epochs $(EPOCHS) -lambda ${LBD} -eta ${ETA} -rank ${RANK} -val \
		$(DATA)/train_${FEAT}_zsh.train.bin.gz $(DATA)/val_${FEAT}_zsh.test.bin.gz \
		$(DATA)/att_${ATT1}_train.bin $(DATA)/att_${ATT1}_val.bin \
		$(DATA)/att_${ATT2}_train.bin $(DATA)/att_${ATT2}_val.bin \
		$(FINAL)/emb_tensor_components_${ATT1}_val \
	>> $@

run_one_test:
	make $(TMP)/msje_lowrank_${RANK}_${ETA}_${NITER}_${FEAT}_${ATT1}${ATT2}_test.txt

$(TMP)/msje_lowrank_${RANK}_${ETA}_${NITER}_${FEAT}_${ATT1}${ATT2}_test.txt:
	mkdir -p $(FINAL)
	$(BIN)/msje_lowrank -epochs ${NITER} -lambda ${LBD} -eta ${ETA} -rank ${RANK} \
		$(DATA)/trainval_${FEAT}_zsh.train.bin.gz $(DATA)/test_${FEAT}_zsh.test.bin.gz \
		$(DATA)/att_${ATT1}_trainval.bin $(DATA)/att_${ATT1}_test.bin \
		$(DATA)/att_${ATT2}_trainval.bin $(DATA)/att_${ATT2}_test.bin \
		$(FINAL)/emb_tensor_components_${ATT1}_test \
	>> $@


#########################################################################
#				SCORING	    			        #
#########################################################################

run_test:
	-rm $(TMP)/results_${ATT1}${ATT2}.txt
	for rank in 100 200 400 600 800 1000 1500 2000; do \
		echo `cat $(TMP)/msje_lowrank_$${rank}_${ETA}_${NITER}_${FEAT}_${ATT1}${ATT2}_val.txt | tail -1` >> $(TMP)/results_${ATT1}${ATT2}.txt; \
	done;
	-rm $(TMP)/param_${ATT1}${ATT2}.txt 
	less $(TMP)/results_${ATT1}${ATT2}.txt | sort -n | tail -1 > $(TMP)/param_${ATT1}${ATT2}.txt
	make run_one_test RANK=`awk '{print $$4}' $(TMP)/param_${ATT1}${ATT2}.txt` NITER=`awk '{print $$3}' $(TMP)/param_${ATT1}${ATT2}.txt`
