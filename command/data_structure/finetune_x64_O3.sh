#!/usr/bin/env bash

CHECKPOINT_PATH=checkpoints/finetune_x64_O3
mkdir -p $CHECKPOINT_PATH
rm -f $CHECKPOINT_PATH/checkpoint_best.pt
cp checkpoints/pretrain/checkpoint_best.pt $CHECKPOINT_PATH/

TOTAL_UPDATES=6000    # Total number of training steps
WARMUP_UPDATES=100    # Warmup the learning rate over this many updates
PEAK_LR=1e-5          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512 # Max sequence length
MAX_POSITIONS=512     # Num. positional embeddings (usually same as above)
MAX_SENTENCES=4       # Number of sequences per batch (batch size)
NUM_CLASSES=44
ENCODER_EMB_DIM=768
ENCODER_LAYERS=8
ENCODER_ATTENTION_HEADS=12

CUDA_VISIBLE_DEVICES=1 python train.py \
  data-bin/data_structure/x64-O3 \
  --num-classes $NUM_CLASSES \
  --user-dir custom_tasks \
  --task data_structure_mf --criterion data_structure_mf --arch roberta_multifield \
  --reset-optimizer --reset-dataloader --reset-meters \
  --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
  --max-positions 512 --max-sentences $MAX_SENTENCES \
  --max-update $TOTAL_UPDATES --update-freq 8 --log-format json --log-interval 10 \
  --no-epoch-checkpoints --save-dir $CHECKPOINT_PATH/ \
  --memory-efficient-fp16 \
  --restore-file $CHECKPOINT_PATH/checkpoint_best.pt |
  tee result/finetune_cf_x64_O3
