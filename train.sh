#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export BERT_PATH=library/full_tf_lm
export DATA_PATH=data/combine
export SEED=555555

python3 train.py \
	--data_dir=${DATA_PATH} \
	--bert_config_file=${BERT_PATH}/roberta-large-config.json \
	--output_dir=checkpoints/run_${SEED} \
	--init_checkpoint=${BERT_PATH}/roberta-large.ckpt \
	--dropout_rate 0.5 \
	--l2_reg_lambda 0.2 \
	--batch_size=16 \
	--max_seq_len=128 \
	--learning_rate=1e-5 \
	--do_train \
	--do_eval \
	--save_checkpoints_steps=5000 \
	--seed ${SEED} \
	--num_train_epochs=10
