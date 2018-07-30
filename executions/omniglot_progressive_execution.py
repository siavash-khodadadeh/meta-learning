import tensorflow as tf

import numpy as np

import settings
from datasets.data_generator import DataGenerator
from models import NeuralNetwork, ProgressiveModelAgnosticMetaLearning

LOG_DIR = settings.BASE_LOG_ADDRESS + '/omniglot/'
TRAIN = True
NUM_CLASSES = 2
UPDATE_BATCH_SIZE = 5
META_BATCH_SIZE = 1
MAML_TRAIN_ITERATIONS = 50001
MAML_ADAPTATION_ITERATIONS = 3
SAVING_PATH = settings.SAVED_MODELS_ADDRESS + '/omniglot'


def train_maml():
    data_generator = DataGenerator(UPDATE_BATCH_SIZE * 2, META_BATCH_SIZE, num_classes=NUM_CLASSES)
    with tf.variable_scope('data_reader'):
        image_tensor, label_tensor = data_generator.make_data_tensor(
            train=TRAIN,
            binary_classification=True
        )

    with tf.variable_scope('train_data'):
        input_data_ph = tf.slice(image_tensor, [0, 0, 0], [-1, NUM_CLASSES * UPDATE_BATCH_SIZE, -1], name='train')
        input_labels_ph = tf.slice(label_tensor, [0, 0], [-1, NUM_CLASSES * UPDATE_BATCH_SIZE], name='labels')
        input_data_ph = tf.reshape(input_data_ph, (-1, 28, 28, 1))
        input_labels_ph = tf.reshape(input_labels_ph, (-1, 1))
        tf.summary.image('train', input_data_ph, max_outputs=25)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.slice(image_tensor, [0, NUM_CLASSES * UPDATE_BATCH_SIZE, 0], [-1, -1, -1], name='validation')
        val_labels_ph = tf.slice(label_tensor, [0, NUM_CLASSES * UPDATE_BATCH_SIZE], [-1, -1], name='val_labels')
        val_data_ph = tf.reshape(val_data_ph, (-1, 28, 28, 1))
        val_labels_ph = tf.reshape(val_labels_ph, (-1, 1))
        tf.summary.image('validation', val_data_ph, max_outputs=25)

    progressive_maml = ProgressiveModelAgnosticMetaLearning(
        NeuralNetwork,
        input_data_ph,
        input_labels_ph,
        val_data_ph,
        val_labels_ph,
        log_dir=LOG_DIR,
        learning_rate=0.1,
        meta_learn_rate=0.00001,
        log_device_placement=False,
        saving_path=SAVING_PATH,
    )
    tf.train.start_queue_runners(progressive_maml.sess)

    if TRAIN:
        print('Start meta training.')

        it = 0
        for it in range(MAML_TRAIN_ITERATIONS):
            progressive_maml.sess.run(progressive_maml.train_op)

            if it % 20 == 0:
                merged_summary = progressive_maml.sess.run(progressive_maml.merged)
                progressive_maml.file_writer.add_summary(merged_summary, global_step=it)
                print(it)
            if it % 2000 == 0:
                progressive_maml.save_model(path=SAVING_PATH, step=it)

        if it != 0:
            progressive_maml.save_model(path=SAVING_PATH, step=it)

    else:
        progressive_maml.load_model(path=SAVING_PATH + '/-50000')
        input_data_np, input_labels_np, val_data_np, val_labels_np = progressive_maml.sess.run(
            (input_data_ph, input_labels_ph, val_data_ph, val_labels_ph)
        )
        progressive_maml.meta_test(num_iterations=1, feed_data={
            progressive_maml.input_data: input_data_np, progressive_maml.input_labels: input_labels_np}
        )
        # progressive_maml.learn_new_concept(instances=input_data_np, labels=input_labels_np, iterations=1)
        outputs = progressive_maml.evaluate(val_data_np)
        # print(outputs)
        # print(val_labels_np)
        print(np.where(outputs[0] > 0.))
        print(np.where(val_labels_np > 0.5))

    print('done')


if __name__ == '__main__':
    train_maml()
