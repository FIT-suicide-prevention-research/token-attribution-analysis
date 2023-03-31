#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py umd/train.parquet.gzip umd/test.parquet.gzip --model_name bert-large-cased --batch_size 16 > log_bert-base-uncased.txt &
P1=$!
CUDA_VISIBLE_DEVICES=1 python train.py umd/train.parquet.gzip umd/test.parquet.gzip --model_name microsoft/deberta-v3-base > log_deberta-v3-base.txt &
P2=$!
CUDA_VISIBLE_DEVICES=2 python train.py umd/train.parquet.gzip umd/test.parquet.gzip --model_name microsoft/deberta-v3-large --batch_size 8 > log_deberta-v3-large.txt &
P3=$!
CUDA_VISIBLE_DEVICES=3 python train.py umd/train.parquet.gzip umd/test.parquet.gzip --model_name xlm-roberta-base > log_xlm-roberta-base.txt &
P4=$!
wait $P1 $P2 $P3 $P4