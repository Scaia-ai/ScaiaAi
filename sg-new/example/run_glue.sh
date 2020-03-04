#!/bin/bash
export TASK_NAME=STS-B

export GLUE_DIR=~/Desktop/Scaia/datasets/glue/glue_data
export CUDA_VISIBLE_DEVICES=0
python run_glue_new.py   --model_type bert   --model_name_or_path bert-base-cased   --task_name $TASK_NAME   --do_train   --do_eval   --do_lower_case   --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128   --per_gpu_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /tmp/$TASK_NAME/
