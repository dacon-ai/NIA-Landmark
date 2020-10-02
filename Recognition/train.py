import functools
import os
import math

from pathlib import Path
from tqdm import tqdm

from absl import app
from absl import flags

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.model import DelfArcFaceModel
from utils.preprocessing import CreateDataset


os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = flags.FLAGS

flags.DEFINE_string('train_directory', '/tmp/', 'Training data directory.')
flags.DEFINE_boolean('generate_train_validation_splits', False,
                     '(Optional) Whether to split the train dataset into'
                     'TRAIN and VALIDATION splits.')
flags.DEFINE_float('validation_split_size', 0.2,
                   '(Optional) The size of the VALIDATION split as a fraction'
                   'of the train dataset.')
flags.DEFINE_integer('seed', 0,
                     '(Optional) The seed to be used while shuffling the train'
                     'dataset when generating the TRAIN and VALIDATION splits.'
                     'Recommended for splits reproducibility purposes.')

strategy = tf.distribute.MirroredStrategy()


EPOCHS = 1000
batch_size = 32
image_size = 224
STEPS_PER_TPU_CALL = 1
learning_rate = 5e-5  # should be smaller than training on single GPU
feature_size = 2048  # Embedding size before the output layer
save_interval = 2000

# ArcFace params
margin = 0.1  # DELG used 0.1, original ArcFace paper used 0.5. When margin is 0, it should be the same as doing a normal softmax but with embedding and weight normalised.
logit_scale = int(math.sqrt(feature_size))

# GeM params
gem_p = 3.
train_p = False  # whether to learn gem_p or not

data_dir = "/home/ubuntu/Dacon/jin/NIA"
checkpoint_dir = "/home/ubuntu/Dacon/jin/NIA/checkpoint/"
train_tf_records_dir = "/home/ubuntu/Dacon/jin/NIA/tfrecords/train*"
validation_tf_records_dir = "/home/ubuntu/Dacon/jin/NIA/tfrecords/validation*"
test_tf_records_dir = "/home/ubuntu/Dacon/jin/NIA/tfrecords_test/test*"

def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model((images, labels), training=True)

        loss = compute_loss(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss * strategy.num_replicas_in_sync)
    train_accuracy(labels, predictions)
    return loss


@tf.function
def distributed_train_steps(training_set_iter, steps_per_call):
    for _ in tf.range(steps_per_call):
        per_replica_losses = strategy.run(train_step, next(training_set_iter))

def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model((images, labels), training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


@tf.function
def distributed_test_step(images, labels):
    return strategy.run(test_step, args=(images, labels,))

training_csv_path = os.path.join(data_dir, "train.csv")
train_csv = pd.read_csv(str(training_csv_path))
num_samples = len(train_csv["id"].tolist())
unique_landmark_ids = train_csv["landmark_id"].unique().tolist()
unique_landmark_ids = tf.convert_to_tensor(unique_landmark_ids, dtype=tf.int64)
training_set = CreateDataset(train_tf_records_dir)
training_set = strategy.experimental_distribute_dataset(training_set)

validation_set = CreateDataset(validation_tf_records_dir)
validation_set = strategy.experimental_distribute_dataset(validation_set)

test_set = CreateDataset(test_tf_records_dir)

train_iter = iter(training_set)


with strategy.scope():
    model = DelfArcFaceModel(
            input_shape=(image_size, image_size, 3), n_classes=len(unique_landmark_ids), margin=margin, logit_scale=logit_scale,
            p=gem_p, train_p=train_p, feature_size=feature_size
        )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        step = 0
        val_step = 0
        hist = []
        with tqdm(total=int(num_samples)) as pbar:
            while True:
                distributed_train_steps(train_iter, tf.convert_to_tensor(STEPS_PER_TPU_CALL))
                template = 'Epoch {}, Training, Loss: {:.4f}, Accuracy: {:.4f}'
                pbar.set_description(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100))
                if step % save_interval == 0:
                    if step == 0:
                        model.summary()
                        print()
                        print("\nlearning rate: {}\nmargin: {}\nlogit_scale: {}\ngem_p: {}\ntrain_p{}\n".format(learning_rate, margin, logit_scale, gem_p, train_p))

                    checkpoint_path = str(os.path.join(checkpoint_dir, "cp_epoch_{}_step_{}".format(epoch, step)))
                    model.save_weights(checkpoint_path)
                    print("Model saved to {}".format(checkpoint_path))
                step += batch_size * STEPS_PER_TPU_CALL
                pbar.update(batch_size * STEPS_PER_TPU_CALL)
                if step >= int(num_samples):
                    break

        with tqdm(total=int(num_samples)*0.2) as pbar:
            for test_images, test_labels in validation_set:
                distributed_test_step(test_images, test_labels)
                template = 'Epoch {}, Validation, Loss: {:.4f}, Accuracy: {:.4f}'
                pbar.set_description(template.format(epoch + 1, test_loss.result(), test_accuracy.result() * 100))
                val_step += batch_size * STEPS_PER_TPU_CALL
                pbar.update(batch_size)
                if val_step >= int(num_samples)*0.2:
                    break

        template = 'Epoch {}, \nTraining Loss: {}, Accuracy: {}\nTest Loss: {}, Accuracy: {}'
        hist.append({"Epoch": epoch,
                     "train_loss": train_loss.result(),
                     "train_accuracy": (train_accuracy.result() * 100),
                     "test_loss": test_loss.result(),
                     "test_accuracy": (test_accuracy.result() * 100)})
        print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))    