#!/usr/bin/env bash

TOTAL_UPDATES=30000 # Total number of training steps
WARMUP_UPDATES=100  # Warmup the learning rate over this many updates
LR=1e-5             # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=4     # Batch size.

CLR_PATH=checkpoints/clr_multifield_any
mkdir -p $CLR_PATH
rm -f $CLR_PATH/checkpoint_best.pt
cp checkpoints/pretrain/checkpoint_best.pt $CLR_PATH/

CUDA_VISIBLE_DEVICES=0 python train.py data-bin/clr_multifield_any \
  --max-positions 512 \
  --max-sentences $MAX_SENTENCES \
  --user-dir custom_tasks \
  --task clr_multifield \
  --reset-optimizer --reset-dataloader --reset-meters \
  --required-batch-size-multiple 1 \
  --arch roberta_multifield \
  --criterion clr_multifield \
  --dropout 0.1 --attention-dropout 0.1 \
  --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
  --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_UPDATES --max-update $TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
  --best-checkpoint-metric AUC --maximize-best-checkpoint-metric \
  --find-unused-parameters \
  --no-epoch-checkpoints --update-freq 8 --log-format=json --log-interval 10 \
  --save-dir $CLR_PATH \
  --memory-efficient-fp16 \
  --pooler-activation-fn relu \
  --truncate-sequence \
  --last-layer -3 \
  --restore-file $CLR_PATH/checkpoint_best.pt | tee result/clr_multifield_any
