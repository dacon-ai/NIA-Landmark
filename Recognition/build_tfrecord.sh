#! /bin/bash

python /home/ubuntu/Dacon/jin/NIA/utils/build_tfrecord.py \
  --train_csv_path=/home/ubuntu/Dacon/jin/NIA/train.csv \
  --train_directory=/home/ubuntu/Dacon/cpt_data/landmark/train/*/ \
  --output_directory=/home/ubuntu/Dacon/jin/NIA/tfrecords \
  --num_shards=128 \
  --generate_train_validation_splits \
  --validation_split_size=0.2