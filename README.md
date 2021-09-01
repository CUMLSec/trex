## Introduction

Trex is a tool to match semantically similar functions based on transfer learning. 

## Installation
We recommend `conda` to setup the environment and install the required packages.

First, create the conda environment,

`conda create -n trex python=3.8 numpy scipy scikit-learn requests`

and activate the conda environment:

`conda activate trex`

Then, install the latest PyTorch (assume you have GPU):

`conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

Enter the trex root directory: e.g., `path/to/trex`, and install trex:

`pip install --editable .`

For large datasets install PyArrow: 

`pip install pyarrow`

For faster training install NVIDIA's apex library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

## Preparation

### Pretrained models:

Create the `checkpoints` and `checkpoints/pretrain` subdirectory in `path/to/trex`

`mkdir checkpoints`, `mkdir checkpoints/pretrain`

Download our [pretrained weight parameters](https://drive.google.com/file/d/1xNcW8r01_J2OTZFh1B0eOG5ikj73zhwe/view?usp=sharing) and put in `checkpoints/pretrain`

### Sample data for finetuning similarity

We provide the sample training/testing files of finetuning in [here](https://drive.google.com/drive/folders/1JHq_O1MiQhKBMnmzOjDgDh1r-TUY-jcl?usp=sharing). Download them ad put in `data-src/similarity`.
If you want to prepare the finetuning data yourself, make sure you follow the format shown in `data-src/similarity`. 

The pipeline of data processing should follow `command/pretrain/prepare_json.py` (read the raw binary, e.g., elf, and obtain the raw bytes for each function and save them in `data-raw/funcbytes`), `command/finetune/prepare_finetune_trace.py` (take `data-raw/funcbytes` as input and generate function code+dummy traces in `data-raw/functraces` for finetuning), and `command/finetune/prepare_finetune.py` (generate the actual finetuning dataset in `data-src/similarity).

We have to binarize the data to make it ready to be trained. To binarize the training data for finetuning, run:

`python command/finetune/preprocess.py`

The binarized training data ready for finetuning (for detecting similarity) will be stored at `data-bin/similarity`

## Training

To finetune the model, run:

`./command/finetune/finetune.sh`

The scripts loads the pretrained weight parameters from `checkpoints/pretrain/` and finetunes the model.

### Sample data for pretraining on micro-traces

We also provide (10K) samples and scripts to demonstrate how to pretrain the model. To binarize the training data for pretraining, run:

`python command/pretrain/preprocess_pretrain_10k.py`

The binarized training data ready for pretraining will be stored at `data-bin/pretrain_10k`

To pretrain the model, run:

`./command/pretrain/pretrain_10k.sh`

The pretrained model will be checkpointed at `checkpoints/pretrain_10k`


## Dataset

We put our dataset [here](https://drive.google.com/drive/folders/1FXlrGiZkch9bnAxlrm43IhYGC3r5NveA?usp=sharing).