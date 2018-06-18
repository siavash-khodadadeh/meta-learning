import os

import tensorflow as tf
import numpy as np

from ucf101_data_generator import get_traditional_dataset
from models import ModelAgnosticMetaLearning, C3DNetwork


BASE_ADDRESS = '/home/siavash/UCF-101/'
LOG_DIR = 'logs/ucf101_transfer_learning/'
TRAIN = True
NUM_CLASSES = 80
CLASS_SAMPLE_SIZE = 1
META_BATCH_SIZE = 1
NUM_GPUS = 10
TRANSFER_LEARNING_ITERATIONS = 1001
BATCH_SPLIT_NUM = 4


def print_accuracy(outputs, labels):
    # Because we have multiple GPUs, outputs will be of the shape N x 1 x N in numpy
    print('outputs:')
    print(outputs)
    outputs_np = np.argmax(outputs, axis=2).reshape(-1, int(NUM_CLASSES * CLASS_SAMPLE_SIZE / BATCH_SPLIT_NUM))
    print(outputs_np)
    print('labels:')
    print(labels)
    labels_np = np.argmax(labels.reshape(-1, NUM_CLASSES * CLASS_SAMPLE_SIZE), axis=1)
    print(labels_np)

    print('accuracy:')
    acc_num = np.sum(outputs_np == labels_np)
    acc = acc_num / (NUM_CLASSES * CLASS_SAMPLE_SIZE)
    print(acc_num)
    print(acc)


def transfer_learn():
    test_actions = sorted(os.listdir(BASE_ADDRESS))[-21:]
    for action in []:
        test_actions.remove(action)

    train_dataset, test_dataset = get_traditional_dataset(
        num_train_actions=80,
        base_address=BASE_ADDRESS,
        test_actions=test_actions,
        class_sample_size=CLASS_SAMPLE_SIZE,
    )

    with tf.variable_scope('train_data'):
        input_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        input_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, 80])
        tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=25)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        val_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, 80])
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
        train=TRAIN
    )

    maml.load_model(path='MAML/sports1m_pretrained.model', load_last_layer=False)
    for it in range(TRANSFER_LEARNING_ITERATIONS):
        data = train_dataset.next_batch(num_classes=80, real_labels=True)
        batch_test_data, batch_test_labels = data['train']
        batch_test_val_data, batch_test_val_labels = data['validation']
        batch_split_size = int(NUM_CLASSES / BATCH_SPLIT_NUM)

        for batch_split_index in range(BATCH_SPLIT_NUM):
            start = batch_split_index * batch_split_size
            end = batch_split_index * batch_split_size + batch_split_size
            test_data = batch_test_data[start:end, :, :, :, :]
            test_labels = batch_test_labels[start:end, :]
            test_val_data = batch_test_val_data[start:end, :, :, :, :]
            test_val_labels = batch_test_val_labels[start:end, :]

            maml.sess.run(maml.inner_train_ops, feed_dict={
                input_data_ph: test_data,
                input_labels_ph: test_labels,
            })

            if it % 50 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                merged_summary = maml.sess.run(maml.merged, feed_dict={
                    input_data_ph: test_data,
                    input_labels_ph: test_labels,
                    val_data_ph: test_val_data,
                    val_labels_ph: test_val_labels,
                }, options=run_options, run_metadata=run_metadata)

                # from tensorflow.python.client import timeline
                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # with open('timeline_01.json', 'w') as f:
                #     f.write(chrome_trace)

                # maml.file_writer.add_run_metadata(run_metadata, 'step%03d' % it, global_step=it)
                maml.file_writer.add_summary(merged_summary, global_step=it)
                print('gradient step: ')
                print(it)

                outputs = maml.sess.run(maml.inner_model_out, feed_dict={
                    maml.input_data: test_val_data,
                    maml.input_labels: test_val_labels,
                })

                print_accuracy(outputs, test_val_labels)


if __name__ == '__main__':
    transfer_learn()
