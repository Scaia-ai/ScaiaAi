#!/bin/bash
export TASK_NAME=STS-B

export GLUE_DIR=~/Desktop/Scaia/datasets/glue/glue_data
export CUDA_VISIBLE_DEVICES=0
python prepare_data.py   --model_type bert   --model_name_or_path bert-base-cased   --task_name $TASK_NAME   --do_lower_case   --data_dir $GLUE_DIR/$TASK_NAME --cache_dir ~/Desktop/Scaia/datasets/glue/cache/   --max_seq_length 128  --output_dir ~/Desktop/Scaia/datasets/glue/output/$TASK_NAME/ --overwrite_output_dir --overwrite_cache
