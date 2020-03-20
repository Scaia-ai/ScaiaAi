#!/bin/bash

conda activate huggingface  # activate huggingface environment 

python run_squad.py --model_type bert --model_name_or_path bbc \
--do_train \
--do_eval \
--version_2_with_negative \
--train_file train-v2.0.json \
--predict_file dev-v2.0.json \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--max_seq_length 384 \
--doc_stride 128 \
--per_gpu_eval_batch_size=8   \
--per_gpu_train_batch_size=8   \
--max_steps 2 \
--output_dir out \
--overwrite_output_dir \
--save_steps 5000


# Fine-Tune Previous Model With dev-test.json
python run_squad.py \
--model_type bert \
--model_name_or_path out \
--do_train \
--version_2_with_negative \
--train_file dev-test.json \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--max_seq_length 384 \
--doc_stride 128 \
--max_steps 2 \
--output_dir out2 \
--overwrite_output_dir \
--save_steps 5000
