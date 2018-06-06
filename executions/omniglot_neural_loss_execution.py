import tensorflow as tf

import numpy as np

from data_generator import DataGenerator
from models import ModelAgnosticMetaLearning, NeuralNetwork


LOG_DIR = '../logs/omniglot_neural_loss/'
TRAIN = False
NUM_CLASSES = 5
UPDATE_BATCH_SIZE = 5
META_BATCH_SIZE = 1
MAML_TRAIN_ITERATIONS = 2001
MAML_ADAPTATION_ITERATIONS = 10

def print_accuracy(outputs, labels):
    # Because we have multiple GPUs, outputs will be of the shape N x 1 x N in numpy
    outputs_np = np.argmax(outputs, axis=2).reshape(-1, 25)
    print(outputs_np)
    labels_np = np.argmax(labels.reshape(-1, 5), axis=1)
    print(labels_np)

    print('accuracy:')
    acc_num = np.sum(outputs_np == labels_np)
    acc = acc_num / (NUM_CLASSES * UPDATE_BATCH_SIZE)
    print(acc_num)
    print(acc)


def train_maml():
    data_generator = DataGenerator(UPDATE_BATCH_SIZE * 2, META_BATCH_SIZE)
    with tf.variable_scope('data_reader'):
        image_tensor, label_tensor = data_generator.make_data_tensor(train=TRAIN)

    with tf.variable_scope('train_data'):
        input_data_ph = tf.slice(image_tensor, [0, 0, 0], [-1, NUM_CLASSES * UPDATE_BATCH_SIZE, -1], name='train')
        input_labels_ph = tf.slice(label_tensor, [0, 0, 0], [-1, NUM_CLASSES * UPDATE_BATCH_SIZE, -1], name='labels')
        input_data_ph = tf.reshape(input_data_ph, (-1, 28, 28, 1))
        input_labels_ph = tf.reshape(input_labels_ph, (-1, 5))
        tf.summary.image('train', input_data_ph, max_outputs=25)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.slice(image_tensor, [0, NUM_CLASSES * UPDATE_BATCH_SIZE, 0], [-1, -1, -1], name='validation')
        val_labels_ph = tf.slice(label_tensor, [0, NUM_CLASSES * UPDATE_BATCH_SIZE, 0], [-1, -1, -1], name='val_labels')
        val_data_ph = tf.reshape(val_data_ph, (-1, 28, 28, 1))
        val_labels_ph = tf.reshape(val_labels_ph, (-1, 5))
        tf.summary.image('validation', val_data_ph, max_outputs=25)

    maml = ModelAgnosticMetaLearning(
        NeuralNetwork,
        input_data_ph,
        input_labels_ph,
        val_data_ph,
        val_labels_ph,
        log_dir=LOG_DIR,
        learning_rate=0.001,
        neural_loss_learning_rate=0.001,
        meta_learn_rate=0.0001,
        learn_the_loss_function=True,
        train=TRAIN,
    )
    tf.train.start_queue_runners(maml.sess)

    if TRAIN:
        print('Start meta training.')

        it = 0
        for it in range(MAML_TRAIN_ITERATIONS):
            maml.sess.run(maml.train_op)
            maml.sess.run(maml.loss_func_op)

            if it % 20 == 0:
                merged_summary, _ = maml.sess.run((maml.merged, maml.train_op))
                maml.file_writer.add_summary(merged_summary, global_step=it)
                print(it)

        if it != 0:
            maml.save_model(path='../saved_models/omniglot_neural_loss/model', step=it)

    else:
        maml.load_model('../saved_models/omniglot_neural_loss/model-2000')
        print('Start testing the network')
        test_batch, test_batch_labels, test_val_batch, test_val_batch_labels = maml.sess.run(
            (maml.input_data, maml.input_labels, maml.input_validation, maml.input_validation_labels)
        )

        for it in range(MAML_ADAPTATION_ITERATIONS):
            maml.sess.run(maml.inner_train_ops, feed_dict={
                maml.input_data: test_batch,
                maml.input_labels: test_batch_labels,
            })

            if it % 1 == 0:
                print(it)
                summary = maml.sess.run(maml.merged, feed_dict={
                    maml.input_data: test_batch,
                    maml.input_labels: test_batch_labels,
                    maml.input_validation: test_val_batch,
                    maml.input_validation_labels: test_val_batch_labels,
                })
                maml.file_writer.add_summary(summary, global_step=it)

        outputs = maml.sess.run(maml.inner_model_out, feed_dict={
            maml.input_data: test_val_batch,
        })

        print_accuracy(outputs, test_val_batch_labels)

    print('done')


if __name__ == '__main__':
    train_maml()
