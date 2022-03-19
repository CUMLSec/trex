#!/usr/bin/env bash

rm checkpoints/pretrain_10k -rf

mkdir -p checkpoints/pretrain_10k

TOTAL_UPDATES=500000  # Total number of training steps
WARMUP_UPDATES=10000  # Warmup the learning rate over this many updates
PEAK_LR=1e-4          # Peak learning rate, adjust as needed, official suggested: 1e-4
TOKENS_PER_SAMPLE=512 # Max sequence length
MAX_POSITIONS=512     # Num. positional embeddings (usually same as above)
MAX_SENTENCES=32      # Number of sequences per batch (batch size)
UPDATE_FREQ=8         # Increase the batch size 32x
ENCODER_EMB_DIM=768
ENCODER_LAYERS=8
ENCODER_ATTENTION_HEADS=12

CUDA_VISIBLE_DEVICES=1 python train.py \
  data-bin/pretrain_10k \
  --task trex --criterion trex \
  --arch trex --sample-break-mode eos --tokens-per-sample $TOKENS_PER_SAMPLE \
  --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
  --max-update $TOTAL_UPDATES --log-format json --log-interval 10 \
  --no-epoch-checkpoints --save-dir checkpoints/pretrain_10k/ \
  --encoder-layers $ENCODER_LAYERS --encoder-embed-dim $ENCODER_EMB_DIM --encoder-attention-heads $ENCODER_ATTENTION_HEADS \
  --random-token-prob 0.5 --mask-prob 0.2 \
  --memory-efficient-fp16 --batch-size-valid 32 \
  --restore-file checkpoints/pretrain_10k/checkpoint_best.pt |
  tee result/pretrain_10k
