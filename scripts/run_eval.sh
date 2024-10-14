#!/bin/bash

# Train models on wiki+flickr dataset.
# For SimCSE baseline, you just need to (1) set new output_dir (2) --framework simcse (3) remove --feature_file
export export CUDA_VISIBLE_DEVICES=0
# SEED=10

# OUT_DIR=result/roberta/mix_flickr/${SEED}/mse
# OUT_DIR=result/mix_flickr/${SEED}/mse
OUT_DIR=result/mix_flickr/mse/1_


# python simcse_to_huggingface.py --path $OUT_DIR

python src/evaluation.py \
      --model_name_or_path $OUT_DIR \
      --pooler cls_before_pooler \
      --task_set sts \
      --mode test
