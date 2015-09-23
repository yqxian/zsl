#!/bin/bash

# Path of the directory that you downloaded the code and the data
PATH_ALL=/BS/Deep_Fragments/work/MSc
DATASET=AWA
DATA=${PATH_ALL}/data/${DATASET}
BIN=${PATH_ALL}/tensor_lowrank
TMP=${PATH_ALL}/tensor_lowrank/results_${DATASET}

# INT = wherever the learned embeddings are going to be saved
INT=${DATA}/SJE_TENSOR

FEAT=goog
last_niter=0
epochs=13
lambda=1e-5
mkdir -p ${TMP};
#After each epoch(around 2 hours), save the tensor to file and report the accuracy on the full test set
	for ATT1 in cont; do \
		for ATT2 in wordnet; do \
			for eta in 1e-3; do \
				for rank in 500; do \
				screen -dmS msje_lowrank_${ATT1}${ATT2}_${rank}_${eta}_${lambda}_${epochs} /bin/bash -c "${BIN}/msje_lowrank -epochs ${epochs} -lambda ${lambda} -rank ${rank} -eta ${eta} \
				${DATA}/trainval_${FEAT}_zsh.train.bin.gz ${DATA}/test_${FEAT}_zsh.test.bin.gz \
				${DATA}/att_${ATT1}_trainval.bin ${DATA}/att_${ATT1}_test.bin \
				${DATA}/att_${ATT2}_trainval.bin ${DATA}/att_${ATT2}_test.bin \
				${TMP}/tensor_${ATT1}${ATT2} \
				>> ${TMP}/msje_lowrank_${epochs}_${rank}_${eta}_${lambda}_${FEAT}_${ATT1}${ATT2}_test.txt;";\
				done;\
			done;\
		done;\
	done
#/BS/Deep_Fragments/work/MSc/tensor_fast/SJE/svm/sje -epochs 1000 -lambda 1e-5 -eta 1e-5 \
#		/BS/Deep_Fragments/work/MSc/tensor_fast/data/CUB/trainval_goog_zsh.train.bin.gz /BS/Deep_Fragments/work/MSc/tensor_fast/data/CUB/test_goog_zsh.test.bin.gz \
#		/BS/Deep_Fragments/work/MSc/CVPR2015/data/CUB/att_${ATT1}_trainval.bin /BS/Deep_Fragments/work/MSc/CVPR2015/data/CUB/att_${ATT1}_test.bin \
#		/BS/Deep_Fragments/work/MSc/CVPR2015/data/CUB/att_${ATT2}_trainval.bin /BS/Deep_Fragments/work/MSc/CVPR2015/data/CUB/att_${ATT2}_test.bin \
#		/BS/Deep_Fragments/work/MSc/tensor_fast/tensor_${ATT1}${ATT2}.bin \
#	>> /BS/Deep_Fragments/work/MSc/tensor_fast/results/sjetensor_fast_1e-5_1e-5_1000_zsh_goog_${ATT1}${ATT2}_test.txt
