import os

import tensorflow as tf
import numpy as np

from ucf101_data_generator import get_fast_dataset
from models import ModelAgnosticMetaLearning, C3DNetwork


BASE_ADDRESS = '/home/siavash/programming/C3D-tensorflow/UCF-101/'
LOG_DIR = 'logs/ucf101/'
TRAIN = True
NUM_CLASSES = 5
UPDATE_BATCH_SIZE = 5
META_BATCH_SIZE = 1


def print_accuracy(outputs, labels):
    outputs_np = np.argmax(outputs, axis=1)
    print(outputs_np)
    labels_np = np.argmax(labels.reshape(-1, 5), axis=1)
    print(labels_np)

    print('accuracy:')
    acc_num = np.sum(outputs_np == labels_np)
    acc = acc_num / len(labels)
    print(acc_num)
    print(acc)


def train_maml():
    train_actions = sorted(os.listdir(BASE_ADDRESS))[:80]
    train_example, val_example = get_fast_dataset(train_actions)

    with tf.variable_scope('train_data'):
        input_data_ph = train_example['video']
        input_labels_ph = train_example['task']
        tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=25)

    with tf.variable_scope('validation_data'):
        val_data_ph = val_example['video']
        val_labels_ph = val_example['task']
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
        # maml.load_model(path='MAML/sports1m_pretrained.model', load_last_layer=False)
        print('start meta training.')

        it = 0
        for it in range(1001):
            maml.sess.run(maml.train_op)

            if it % 20 == 0 and it != 0:
                merged_summary = maml.sess.run(maml.merged)
                maml.file_writer.add_summary(merged_summary, global_step=it)
                print(it)

        if it != 0:
            maml.save_model(path='saved_models/ucf101/model', step=it)

    else:
        test_actions = sorted(os.listdir(BASE_ADDRESS))[80:]
        test_example, test_val_example = get_fast_dataset(test_actions)
        maml.load_model(path='saved_models/backups/ucf101/model-1000')
        print('Start testing the network')

        test_data, test_labels, test_val_data, test_val_labels = maml.sess.run(
            (test_example['video'], test_example['task'], test_val_example['video'], test_val_example['task'])
        )

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

                outputs, loss = maml.sess.run([maml.model_out_train, maml.train_loss], feed_dict={
                    maml.input_data: test_val_data,
                    maml.input_labels: test_val_labels,
                })

                print_accuracy(outputs, test_val_labels)


if __name__ == '__main__':
    train_maml()
