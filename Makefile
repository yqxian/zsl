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
DATA=$(PATH_ALL)/data/$(DATASET)
BIN=$(PATH_ALL)/SJE_regularized/bin
TMP=$(PATH_ALL)/SJE_regularized/results_$(DATASET)

# INT = wherever the learned embeddings are going to be saved
FINAL=$(DATA)/SJE_REGULARIZED/MAT_$(ETA)_$(LBD)_$(NITER)_$(ATT)
EPOCHS=15

#########################################################################
#			SJE ZERO_SHOT 					#
#########################################################################
CUB:
	-rm -rf $(PATH_ALL)/data/CUB/SJE_REGULARIZED
	make run_dataset DATASET=CUB

AWA:
	-rm -rf $(PATH_ALL)/data/AWA/SJE_REGULARIZED
	make run_dataset DATASET=AWA

Dogs:
	-rm -rf $(PATH_ALL)/data/Dogs113/SJE_REGULARIZED
	make run_dataset DATASET=Dogs113	

#screen -S sje_${DATASET}_$${att} -m -d make sje_zsh ATT=$${att};
run_all:
	for att in cont word2vec glove wordnet; do \
		screen -S sje_${DATASET}_$${att} -m -d make sje_zsh ATT=$${att};\
	done

run_all2:
	for att in bow word2vec glove wordnet; do \
		screen -S sje_${DATASET}_$${att} -m -d make sje_zsh ATT=$${att};\
	done

run_dataset:
	-rm -r $(TMP)
	mkdir -p $(TMP)
ifeq ($(DATASET), Dogs113)
	make run_all2
else
	make run_all
endif

sje_zsh:
	for lbd in 100 10 1 0 1e-1 1e-2 1e-3 1e-4 1e-5; do \
		make sje_one_val NITER=$(EPOCHS) FEAT=goog LBD=$${lbd} ETA=0.1;\
	done
	make sje_test FEAT=goog NITER=$(EPOCHS) ETA=0.1

sje_one_val: 
	make $(TMP)/sje_${ETA}_${LBD}_${NITER}_${FEAT}_${ATT}_val.txt

$(TMP)/sje_${ETA}_${LBD}_${NITER}_${FEAT}_${ATT}_val.txt:
	mkdir -p $(FINAL)
	$(BIN)/sje_regularized -epochs ${NITER} -eta ${ETA} -lambda ${LBD} -val \
		$(DATA)/train_${FEAT}_zsh.train.bin.gz $(DATA)/val_${FEAT}_zsh.test.bin.gz \
		$(DATA)/att_${ATT}_train.bin $(DATA)/att_${ATT}_val.bin \
		$(FINAL)/emb_mat_${ATT}_val \
	>> $@

sje_one_test:
	make $(TMP)/sje_${ETA}_${LBD}_${NITER}_${FEAT}_${ATT}_test.txt

$(TMP)/sje_${ETA}_${LBD}_${NITER}_${FEAT}_${ATT}_test.txt:
	mkdir -p $(FINAL)
	$(BIN)/sje_regularized -epochs ${NITER} -eta ${ETA} -lambda ${LBD} \
		$(DATA)/trainval_${FEAT}_zsh.train.bin.gz $(DATA)/test_${FEAT}_zsh.test.bin.gz \
		$(DATA)/att_${ATT}_trainval.bin $(DATA)/att_${ATT}_test.bin \
		$(FINAL)/emb_mat_${ATT}_test \
	>> $@

#########################################################################
#				SCORING	    			        #
#########################################################################

sje_test:
	-rm ${TMP}/sje_results_${ATT}.txt;\
	for lbd in 100 10 1 0 1e-1 1e-2 1e-3 1e-4 1e-5; do \
		echo `cat $(TMP)/sje_${ETA}_$${lbd}_${NITER}_${FEAT}_${ATT}_val.txt | tail -1` >> $(TMP)/sje_results_${ATT}.txt;\
	done;	
	-rm $(TMP)/param_${ATT}.txt
	less $(TMP)/sje_results_${ATT}.txt | sort -n | tail -1 > $(TMP)/param_${ATT}.txt
	make sje_one_test LBD=`awk '{print $$3}' $(TMP)/param_${ATT}.txt` NITER=`awk '{print $$4}' $(TMP)/param_${ATT}.txt`
