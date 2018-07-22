import os

import tensorflow as tf
import numpy as np
import random

from ucf101_data_generator import get_traditional_dataset
from models import ModelAgnosticMetaLearning, C3DNetwork


BASE_ADDRESS = '/home/siavash/UCF-101/'
LOG_DIR = 'logs/ucf101_transfer_learning_80_to_5/'
TRAIN = True
NUM_CLASSES = 5
CLASS_SAMPLE_SIZE = 1
META_BATCH_SIZE = 1
NUM_GPUS = 1
TRANSFER_LEARNING_ITERATIONS = 401
BATCH_SPLIT_NUM = 1


random.seed(100)
tf.set_random_seed(100)


def print_accuracy(outputs, labels):
    # Because we have multiple GPUs, outputs will be of the shape N x 1 x N in numpy
    # print('outputs:')
    # print(outputs)
    outputs_np = np.argmax(outputs, axis=2).reshape(-1, int(NUM_CLASSES * CLASS_SAMPLE_SIZE / BATCH_SPLIT_NUM))
    # print(outputs_np)
    # print('labels:')
    # print(labels)
    labels_np = np.argmax(labels.reshape(-1, NUM_CLASSES * CLASS_SAMPLE_SIZE), axis=1)
    # print(labels_np)

    # print('accuracy:')
    acc_num = np.sum(outputs_np == labels_np)
    acc = acc_num / int(NUM_CLASSES * CLASS_SAMPLE_SIZE / BATCH_SPLIT_NUM)
    # print(acc_num)
    # print(acc)
    return acc


def transfer_learn():
    test_actions = [
        'CleanAndJerk',
        'MoppingFloor',
        'FrontCrawl',
        'Surfing',  # 0
        'Bowling',
        'SoccerPenalty',
        'SumoWrestling',
        'Shotput',
        'PlayingSitar',
        'FloorGymnastics',
        'Typing',  # 1
        'JumpingJack',
        'ShavingBeard',
        'FrisbeeCatch',
        'WritingOnBoard',
        'JavelinThrow',
        'Fencing',
        'FieldHockeyPenalty',  # 3
        'BaseballPitch',  # 4
        'CuttingInKitchen',
        'Kayaking',  # 2
    ]

    random.shuffle(test_actions)
    test_actions = test_actions[:5]

    train_dataset, test_dataset = get_traditional_dataset(
        base_address=BASE_ADDRESS,
        class_sample_size=CLASS_SAMPLE_SIZE,
        test_actions=test_actions
    )

    with tf.variable_scope('train_data'):
        input_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        input_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=25)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        val_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5])
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

    maml.load_model(path='saved_models/transfer_learning/model-200', load_last_layer=False)

    data = test_dataset.next_batch(num_classes=5)
    test_data, test_labels = data['train']
    test_val_data, test_val_labels = data['validation']

    for it in range(TRANSFER_LEARNING_ITERATIONS):
        print(it)
        if it % 50 == 0:
            if it % 100 == 0:
                maml.save_model('saved_models/transfer_learning_80_5/model', step=it)

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            merged_summary = maml.sess.run(maml.merged, feed_dict={
                input_data_ph: test_data,
                input_labels_ph: test_labels,
                val_data_ph: test_val_data,
                val_labels_ph: test_val_labels,
            }, options=run_options, run_metadata=run_metadata)
            maml.file_writer.add_summary(merged_summary, global_step=it)

            outputs = maml.sess.run(maml.inner_model_out, feed_dict={
                maml.input_data: test_val_data,
                maml.input_labels: test_val_labels,
            })

            val_acc = print_accuracy(outputs, test_val_labels)

            print('iteration: {}'.format(it))
            print('Validation accuracy on all batches: ')
            print(val_acc)

        maml.sess.run(maml.inner_train_ops, feed_dict={
            input_data_ph: test_data,
            input_labels_ph: test_labels,
        })

    maml.save_model('saved_models/transfer_learning_80_5/model', step=it)


if __name__ == '__main__':
    transfer_learn()
