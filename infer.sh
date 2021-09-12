#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

BERT_PATH=library/full_tf_lm
DATA_PATH=data/evaluation_data
CKPT_PATH='checkpoints/run_234567/'

for i in "dailydialog" "persona" "empathetic" "topical" "movie" "twitter"; do
    echo "$i"
	python3 eval.py --data_dir=${DATA_PATH} --bert_config_file=${BERT_PATH}/roberta-large-config.json --eval_file_path=${DATA_PATH}/${i}_test.txt --output_dir=${CKPT_PATH} \
	    --init_checkpoint=${BERT_PATH}/roberta-large.ckpt --batch_size=16 --max_seq_len=128 --do_evaluate
done ;
mv ${DATA_PATH}/*.score ${CKPT_PATH}
mv ${DATA_PATH}/*.tf_record ${CKPT_PATH}
