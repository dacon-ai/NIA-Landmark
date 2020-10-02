import tensorflow as tf
from tqdm import tqdm
import os
from pathlib import Path
import math
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.model import *
from utils.preprocessing import CreateDataset
os.environ["CUDA_VISIBLE_DEVICES"]="1"

batch_size = 64
image_size = 224
learning_rate = 5e-5  # should be smaller than training on single GPU
feature_size = 2048  # Embedding size before the output layer

# ArcFace params
margin = 0.1  # DELG used 0.1, original ArcFace paper used 0.5. When margin is 0, it should be the same as doing a normal softmax but with embedding and weight normalised.
logit_scale = int(math.sqrt(feature_size))

# GeM params
gem_p = 3.
train_p = False  # whether to learn gem_p or not

data_dirs = '/home/ubuntu/Dacon/cpt_data/landmark/'
data_dir = "/home/ubuntu/Dacon/jin/NIA"

checkpoint_dir = "/home/ubuntu/Dacon/jin/NIA/checkpoint/"
train_tf_records_dir = "/home/ubuntu/Dacon/jin/NIA/tfrecords/train*"
test_tf_records_dir = "/home/ubuntu/Dacon/jin/NIA/tfrecords/validation*"
training_csv_path = os.path.join(data_dir, "train.csv")
train_csv = pd.read_csv(str(training_csv_path))
num_samples = len(train_csv["id"].tolist())
unique_landmark_ids = train_csv["landmark_id"].unique().tolist()
unique_landmark_ids = tf.convert_to_tensor(unique_landmark_ids, dtype=tf.int64)

test_tf_records_dir = "/home/ubuntu/Dacon/jin/NIA/tfrecords_test/train*"
test_set = CreateDataset(test_tf_records_dir, batch_size=batch_size, augmentation=True)

new_model = tf.keras.models.load_model('checkpoint/cp_epoch_2/')

eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='eval_accuracy')
eval_accuracy.reset_states()

with tqdm(total=5000) as pbar:
    step = 0
    template = 'Accuracy: {:.4f}'
    for images, labels in test_set:
      eval_step(images, labels)
      pbar.set_description(template.format(eval_accuracy.result() * 100))
      pbar.update(batch_size)
      step += batch_size
      if step >= int(5000):
        break
    print ('전략을 사용하지 않고, 저장된 모델을 복원한 후의 정확도: {}'.format(eval_accuracy.result()*100))
