#! /bin/bash

python /NIA-Landmark/Recognition/utils/build_tfrecord.py \
  --train_csv_path=/NIA-Landmark/Recognition/train.csv \
  --test_csv_path=/NIA-Landmark/Recognition/test.csv \
  --train_directory=/tmp/train/*/ \
  --test_directory=/tmp/test/*/ \
  --output_directory=/NIA-Landmark/Recognition/tfrecords \
  --num_shards=128 \
  --generate_train_validation_splits \
  --validation_split_size=0.2
