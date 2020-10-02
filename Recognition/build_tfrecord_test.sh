#! /bin/bash

python /home/ubuntu/Dacon/jin/NIA/utils/build_tfrecord.py \
  --test_csv_path=/home/ubuntu/Dacon/jin/NIA/test.csv \
  --test_directory=/home/ubuntu/Dacon/cpt_data/landmark/test/*/ \
  --output_directory=/home/ubuntu/Dacon/jin/NIA/tfrecords_test \
  --num_shards=128 \