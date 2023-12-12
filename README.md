# Learned Go Decompilation with LLMs

This repository contains all the code needed to train and evaluate the models featured in the paper, as well as the Go-L dataset

## Directory Structure

### Toplevel files
`batch.sh`, `dev`, `devrc`: scripts to help with development on HiPerGator

`btc_reproduced.log`: output of BTC Go-S evaluation, resulting in 0.74 AED

`environment-btc.yml`, `environment.yml`: conda environment specifications
Use `conda env create -f environment.yml` to use.

`nix.conf`: configuration to use nix on HiPerGator
Use the following nixpkgs revision: `flake:nixpkgs github:NixOS/nixpkgs/ec750fd01963ab6b20ee1f0cb488754e8036d89d`

### btc\_processing
Contains scripts used to preprocess the BTC dataset

### checkpoint-19128
Final training checkpoint. Use by applying to CodeLLAMA-7B-hf

### eval
Contains tools for evaluating output

`eval.py`: script to generate outputs for all checkpoints

`checkpoint0.json`, `checkpoint1.json`, ... : outputs generated using `eval.py` script

`eval_aed.py`: script to run AED evaluations on `checkpoint.json`s

### fetch
Contains code for creating Go-L dataset

### train
`train_accel.py`, `train_rl.py`, `train_datset.py`: old training scripts

`train_actual.py`: training script used to create final model

`train.json`, `test.json`: training and test datasets
