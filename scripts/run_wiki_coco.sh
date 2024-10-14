#!/bin/bash

# Train models on wiki+coco dataset.
# For SimCSE baseline, you just need to (1) set new output_dir (2) --framework simcse (3) remove --feature_file

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
IMG=data/train_coco_ViT_L14.json
CAPTION=data/coco_random_captions.txt
TEXT=data/wiki1m_for_simcse.txt
# IMAGE_ROOT=data/flickr30k/flickr30k-images

SEED=1
MODEL=Model/bert-base-uncased
FIRST_TEACHER_MODEL=Model/simcse
SECOND_TEACHER_MODEL=Model/DiffCSE
LR=1e-5
BATCH=32
EPS=3
LBD=0.01
MARGIN1=0.18
MARGIN2=0.18
SCORE_BASE=0.66

OUT_DIR=result/mix_coco/best/${SEED}/mse


python src/train_mix.py \
    --framework mse \
    --model_name_or_path $MODEL \
    --text_file $TEXT \
    --caption_file $CAPTION  \
    --feature_file $IMG \
    --output_dir $OUT_DIR \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH \
    --num_train_epochs $EPS \
    --seed $SEED  \
    --margin1 $MARGIN1 \
    --margin2 $MARGIN2 \
    --score_base $SCORE_BASE \
    --lbd $LBD ${@:5} \
    --first_teacher_model_name_or_path $FIRST_TEACHER_MODEL \
    --second_teacher_model_name_or_path $SECOND_TEACHER_MODEL \
    --distillation_loss listmle \
    --alpha_ 0.33 \
    --beta_ 1.0 \
    --gamma_ 1.0 \
    --tau2 0.05 \
    --fp16 \


