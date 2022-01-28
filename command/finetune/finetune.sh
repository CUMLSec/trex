#!/usr/bin/env bash

TOTAL_UPDATES=300000 # Total number of training steps
WARMUP_UPDATES=1000  # Warmup the learning rate over this many updates
LR=1e-5              # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=8      # Batch size.
ENCODER_EMB_DIM=768
ENCODER_LAYERS=8
ENCODER_ATTENTION_HEADS=12

SIMILARITY_PATH=checkpoints/similarity
mkdir -p $SIMILARITY_PATH
rm -f $SIMILARITY_PATH/checkpoint_best.pt
cp checkpoints/pretrain/checkpoint_best.pt $SIMILARITY_PATH/

CUDA_VISIBLE_DEVICES=3 python train.py data-bin/similarity \
  --max-positions 512 \
  --max-sentences $MAX_SENTENCES \
  --task similarity \
  --reset-optimizer --reset-dataloader --reset-meters \
  --required-batch-size-multiple 1 \
  --arch trex \
  --criterion similarity \
  --dropout 0.1 --attention-dropout 0.1 \
  --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
  --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_UPDATES --max-update $TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
  --encoder-layers $ENCODER_LAYERS --encoder-embed-dim $ENCODER_EMB_DIM --encoder-attention-heads $ENCODER_ATTENTION_HEADS \
  --best-checkpoint-metric F1_pair --maximize-best-checkpoint-metric \
  --find-unused-parameters \
  --no-epoch-checkpoints --update-freq 4 --log-format=json --log-interval 10 --max-sentences-valid 32 \
  --save-dir $SIMILARITY_PATH \
  --memory-efficient-fp16 \
  --restore-file $SIMILARITY_PATH/checkpoint_best.pt | tee result/similarity
