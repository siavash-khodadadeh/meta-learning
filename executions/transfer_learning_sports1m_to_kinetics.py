import os

import tensorflow as tf
import numpy as np
import random

from ucf101_data_generator import get_traditional_dataset
from models import ModelAgnosticMetaLearning, C3DNetwork


BASE_ADDRESS = '/home/mkhan/kinetics_dataset2/clips/dataset/train/'
LOG_DIR = 'logs/ucf101_transfer_learning_sports1m_to_kinetics/'
TRAIN = True
NUM_CLASSES = 400
CLASS_SAMPLE_SIZE = 1
META_BATCH_SIZE = 1
BATCH_SIZE = 100
NUM_GPUS = 10
TRANSFER_LEARNING_ITERATIONS = 401


random.seed(100)
tf.set_random_seed(100)


def print_accuracy(outputs, labels):
    # Because we have multiple GPUs, outputs will be of the shape N x 1 x N in numpy
    # print('outputs:')
    # print(outputs)
    outputs_np = np.argmax(outputs, axis=2).reshape(-1, BATCH_SIZE)
    # print(outputs_np)
    # print('labels:')
    # print(labels)
    labels_np = np.argmax(labels.reshape(-1, BATCH_SIZE), axis=1)
    # print(labels_np)

    # print('accuracy:')
    acc_num = np.sum(outputs_np == labels_np)
    acc = acc_num / float(BATCH_SIZE)
    # print(acc_num)
    # print(acc)
    return acc


def transfer_learn():
    train_dataset, test_dataset = get_traditional_dataset(
        base_address=BASE_ADDRESS,
        class_sample_size=CLASS_SAMPLE_SIZE,
        num_train_actions=400
    )

    with tf.variable_scope('train_data'):
        input_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        input_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, 400])
        tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=25)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        val_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, 400])
        tf.summary.image('validation', val_data_ph[:, 0, :, :, :], max_outputs=25)

    gpu_devices = ['/gpu:{}'.format(gpu_id) for gpu_id in range(NUM_GPUS)]

    maml = ModelAgnosticMetaLearning(
        C3DNetwork,
        input_data_ph,
        input_labels_ph,
        val_data_ph,
        val_labels_ph,
        log_dir=LOG_DIR,
        gpu_devices=gpu_devices,
        meta_learn_rate=0.00001,
        learning_rate=0.001,
        train=TRAIN,
        log_device_placement=False
    )

    maml.load_model(path='MAML/sports1m_pretrained.model', load_last_layer=False)

    for it in range(TRANSFER_LEARNING_ITERATIONS):
        print(it)

        data, labels = train_dataset.next_simple_batch(batch_size=BATCH_SIZE)

        if it % 100 == 0:
            maml.save_model('saved_models/transfer_learning/model', step=it)

        if it % 20 == 0:
            merged_summary = maml.sess.run(maml.merged, feed_dict={
                input_data_ph: data,
                input_labels_ph: labels,
                val_data_ph: data,
                val_labels_ph: labels,
            })
            maml.file_writer.add_summary(merged_summary, global_step=it)

            outputs = maml.sess.run(maml.inner_model_out, feed_dict={
                maml.input_data: data,
                maml.input_labels: labels,
            })

            print_accuracy(outputs, labels)

        maml.sess.run(maml.inner_train_ops, feed_dict={
            input_data_ph: data,
            input_labels_ph: labels,
        })


if __name__ == '__main__':
    transfer_learn()
