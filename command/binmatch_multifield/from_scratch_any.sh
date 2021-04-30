#!/usr/bin/env bash

TOTAL_UPDATES=30000 # Total number of training steps
WARMUP_UPDATES=100  # Warmup the learning rate over this many updates
LR=1e-5             # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=1     # Batch size.
UPDATE_FREQ=32
ENCODER_EMB_DIM=512
ENCODER_LAYERS=12
ENCODER_ATTENTION_HEADS=8

BMT_PATH=checkpoints/binmatch_multifield_from_scratch
mkdir -p $BMT_PATH
# cp checkpoints/pretrain/checkpoint_best.pt $CLR_PATH/

python train.py data-bin/binmatch_multifield \
  --max-positions 1536 \
  --max-sentences $MAX_SENTENCES \
  --user-dir custom_tasks \
  --task binmatch_multifield \
  --reset-optimizer --reset-dataloader --reset-meters \
  --required-batch-size-multiple 1 \
  --arch roberta_multifield \
  --criterion binmatch_multifield \
  --dropout 0.1 --attention-dropout 0.1 \
  --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
  --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_UPDATES --max-update $TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
  --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
  --find-unused-parameters \
  --no-epoch-checkpoints --update-freq $UPDATE_FREQ --log-format=json --log-interval 300 \
  --encoder-layers $ENCODER_LAYERS --encoder-embed-dim $ENCODER_EMB_DIM --encoder-attention-heads $ENCODER_ATTENTION_HEADS \
  --save-dir $BMT_PATH \
  --memory-efficient-fp16 \
  --pooler-activation-fn relu \
  --truncate-sequence \
  --restore-file $BMT_PATH/checkpoint_last.pt \
  --num-classes 2 | tee result/binmatch_from_scratch