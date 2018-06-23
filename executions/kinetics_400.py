import os
import random

import tensorflow as tf
import numpy as np

from ucf101_data_generator import get_traditional_dataset
from models import ModelAgnosticMetaLearning, C3DNetwork


BASE_ADDRESS = '/home/mkhan/kinetics_dataset2/clips/dataset/train/'

LOG_DIR = 'logs/kinetics_400/'
TRAIN = True
NUM_CLASSES = 20
CLASS_SAMPLE_SIZE = 1
META_BATCH_SIZE = 1
NUM_GPUS = 10


random.seed(100)
tf.set_random_seed(100)


def print_accuracy(outputs, labels):
    # Because we have multiple GPUs, outputs will be of the shape N x 1 x N in numpy
    print('outputs:')
    print(outputs)
    outputs_np = np.argmax(outputs, axis=2).reshape(-1, NUM_CLASSES * CLASS_SAMPLE_SIZE)
    print(outputs_np)
    print('labels:')
    print(labels)
    labels_np = np.argmax(labels.reshape(-1, NUM_CLASSES), axis=1)
    print(labels_np)

    print('accuracy:')
    acc_num = np.sum(outputs_np == labels_np)
    acc = acc_num / (NUM_CLASSES * CLASS_SAMPLE_SIZE)
    print(acc_num)
    print(acc)


def train_maml():
    test_actions = sorted(os.listdir(BASE_ADDRESS))[-21:]
    for action in []:
        test_actions.remove(action)

    train_dataset, test_dataset = get_traditional_dataset(
        num_train_actions=600,
        base_address=BASE_ADDRESS,
        class_sample_size=CLASS_SAMPLE_SIZE,
    )

    with tf.variable_scope('train_data'):
        input_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        input_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES])
        tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=NUM_CLASSES)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        val_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES])
        tf.summary.image('validation', val_data_ph[:, 0, :, :, :], max_outputs=NUM_CLASSES)

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
        train=TRAIN
    )

    if TRAIN:
        maml.load_model(path='MAML/sports1m_pretrained.model', load_last_layer=False)
        print('start meta training.')

        it = 0
        for it in range(10001):
            data = train_dataset.next_batch(num_classes=NUM_CLASSES)
            tr_data, tr_labels = data['train']
            val_data, val_labels = data['validation']

            if it % 20 == 0:
                merged_summary = maml.sess.run(maml.merged, feed_dict={
                    input_data_ph: tr_data,
                    input_labels_ph: tr_labels,
                    val_data_ph: val_data,
                    val_labels_ph: val_labels,
                })
                maml.file_writer.add_summary(merged_summary, global_step=it)
                print(it)

            maml.sess.run(maml.train_op, feed_dict={
                input_data_ph: tr_data,
                input_labels_ph: tr_labels,
                val_data_ph: val_data,
                val_labels_ph: val_labels,
            })

            if it % 100 == 0:
                maml.save_model(path='saved_models/kinetics400/model', step=it)

        if it != 0:
            maml.save_model(path='saved_models/kinetics400/model', step=it)

    else:
        random.shuffle(test_actions)
        test_actions = test_actions
        train_dataset, test_dataset = get_traditional_dataset(
            base_address='/home/siavash/UCF-101/',
            class_sample_size=CLASS_SAMPLE_SIZE,
            test_actions=test_actions
        )

        maml.load_model(path='saved_models/kinetics400/model-10000')
        print('Start testing the network')
        data = test_dataset.next_batch(num_classes=NUM_CLASSES)
        test_data, test_labels = data['train']
        test_val_data, test_val_labels = data['validation']

        for it in range(5):
            maml.sess.run(maml.inner_train_ops, feed_dict={
                input_data_ph: test_data,
                input_labels_ph: test_labels,
            })

            if it % 1 == 0:
                merged_summary = maml.sess.run(maml.merged, feed_dict={
                    input_data_ph: test_data,
                    input_labels_ph: test_labels,
                    val_data_ph: test_val_data,
                    val_labels_ph: test_val_labels,
                })
                maml.file_writer.add_summary(merged_summary, global_step=it)
                print('gradient step: ')
                print(it)

                outputs = maml.sess.run(maml.inner_model_out, feed_dict={
                    maml.input_data: test_val_data,
                    maml.input_labels: test_val_labels,
                })

                print_accuracy(outputs, test_val_labels)

        maml.save_model('saved_models/ucf101-fit/model-kinetics-trained', step=it)


if __name__ == '__main__':
    train_maml()
