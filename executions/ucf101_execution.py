import os

import tensorflow as tf
import numpy as np

from ucf101_data_generator import get_traditional_dataset
from models import ModelAgnosticMetaLearning, C3DNetwork


BASE_ADDRESS = '/home/siavash/UCF-101/'
LOG_DIR = 'logs/ucf101/'
TRAIN = False
NUM_CLASSES = 5
CLASS_SAMPLE_SIZE = 1
META_BATCH_SIZE = 1


def print_accuracy(outputs, labels):
    # Because we have multiple GPUs, outputs will be of the shape N x 1 x N in numpy
    outputs_np = np.argmax(outputs, axis=2).reshape(-1, 5)
    print(outputs_np)
    labels_np = np.argmax(labels.reshape(-1, 5), axis=1)
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
        num_train_actions=80,
        test_actions=test_actions,
        class_sample_size=CLASS_SAMPLE_SIZE,
    )

    with tf.variable_scope('train_data'):
        input_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        input_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=25)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        val_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        tf.summary.image('validation', val_data_ph[:, 0, :, :, :], max_outputs=25)

    maml = ModelAgnosticMetaLearning(
        C3DNetwork,
        input_data_ph,
        input_labels_ph,
        val_data_ph,
        val_labels_ph,
        log_dir=LOG_DIR,
        meta_learn_rate=0.00001,
        learning_rate=0.001,
        train=TRAIN
    )

    if TRAIN:
        maml.load_model(path='MAML/sports1m_pretrained.model', load_last_layer=False)
        print('start meta training.')

        it = 0
        for it in range(1001):
            data = train_dataset.next_batch(num_classes=5)
            tr_data, tr_labels = data['train']
            val_data, val_labels = data['validation']

            # tr_data = tr_data[::5, :, :, :, :]
            # tr_labels = tr_labels[::5, :]
            # val_data = val_data[::5, :, :, :, :]
            # val_labels = val_labels[::5, :]

            maml.sess.run(maml.train_op, feed_dict={
                input_data_ph: tr_data,
                input_labels_ph: tr_labels,
                val_data_ph: val_data,
                val_labels_ph: val_labels,
            })

            if it % 20 == 0:
                merged_summary = maml.sess.run(maml.merged, feed_dict={
                    input_data_ph: tr_data,
                    input_labels_ph: tr_labels,
                    val_data_ph: val_data,
                    val_labels_ph: val_labels,
                })
                maml.file_writer.add_summary(merged_summary, global_step=it)
                print(it)

        if it != 0:
            maml.save_model(path='saved_models/ucf101/model', step=it)

    else:
        maml.load_model(path='saved_models/ucf101/model-1000')
        print('Start testing the network')
        data = test_dataset.next_batch(num_classes=5)
        print(test_dataset.actions[:5])
        test_data, test_labels = data['train']
        test_val_data, test_val_labels = data['validation']

        # test_data = test_data[::5, :, :, :, :]
        # test_labels = test_labels[::5, :]
        # test_val_data = test_val_data[::5, :, :, :, :]
        # test_val_labels = test_val_labels[::5, :]

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
                print(it)

                outputs = maml.sess.run(maml.inner_model_out, feed_dict={
                    maml.input_data: test_val_data,
                    maml.input_labels: test_val_labels,
                })

                print_accuracy(outputs, test_val_labels)


if __name__ == '__main__':
    train_maml()
