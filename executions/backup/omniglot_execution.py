import tensorflow as tf

import numpy as np

import settings
from datasets.data_generator import DataGenerator
from models import ModelAgnosticMetaLearning, NeuralNetwork

LOG_DIR = settings.BASE_LOG_ADDRESS + '/omniglot/'
TRAIN = True
NUM_CLASSES = 5
UPDATE_BATCH_SIZE = 1
META_BATCH_SIZE = 1
MAML_TRAIN_ITERATIONS = 20001
MAML_ADAPTATION_ITERATIONS = 3
SAVING_PATH = settings.SAVED_MODELS_ADDRESS + '/omniglot'


def train_maml():
    data_generator = DataGenerator(UPDATE_BATCH_SIZE * 2, META_BATCH_SIZE)
    with tf.variable_scope('data_reader'):
        image_tensor, label_tensor = data_generator.make_data_tensor(
            train=TRAIN,
        )

    with tf.variable_scope('train_data'):
        input_data_ph = tf.slice(image_tensor, [0, 0, 0], [-1, NUM_CLASSES * UPDATE_BATCH_SIZE, -1], name='train')
        input_labels_ph = tf.slice(label_tensor, [0, 0, 0], [-1, NUM_CLASSES * UPDATE_BATCH_SIZE, -1], name='labels')
        input_data_ph = tf.reshape(input_data_ph, (-1, 28, 28, 1))
        input_labels_ph = tf.reshape(input_labels_ph, (-1, NUM_CLASSES))
        tf.summary.image('train', input_data_ph, max_outputs=25)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.slice(image_tensor, [0, NUM_CLASSES * UPDATE_BATCH_SIZE, 0], [-1, -1, -1], name='validation')
        val_labels_ph = tf.slice(label_tensor, [0, NUM_CLASSES * UPDATE_BATCH_SIZE, 0], [-1, -1, -1], name='val_labels')
        val_data_ph = tf.reshape(val_data_ph, (-1, 28, 28, 1))
        val_labels_ph = tf.reshape(val_labels_ph, (-1, NUM_CLASSES))

        tf.summary.image('validation', val_data_ph, max_outputs=25)

    # input_data_ph, input_labels_ph, val_data_ph, val_labels_ph, iterator, table = \
    #     get_omniglot_tf_record_dataset(
    #         num_classes=NUM_CLASSES,
    #         num_samples_per_class=1,
    #         meta_batch_size=1,
    #     )

    maml = ModelAgnosticMetaLearning(
        NeuralNetwork,
        input_data_ph,
        input_labels_ph,
        val_data_ph,
        val_labels_ph,
        log_dir=LOG_DIR,
        learning_rate=0.1,
        meta_learn_rate=0.001,
        num_gpu_devices=1,
        num_classes=NUM_CLASSES,
        log_device_placement=False,
        saving_path=SAVING_PATH,
    )
    tf.train.start_queue_runners(maml.sess)
    maml.sess.run(tf.tables_initializer())
    # maml.sess.run(iterator.initializer)

    if TRAIN:
        print('Start meta training.')

        maml.meta_train(num_iterations=MAML_TRAIN_ITERATIONS, report_after_x_step=100, save_after_x_step=20000)
        # it = 0
        # for it in range(MAML_TRAIN_ITERATIONS):
        #     _, merged_summary = maml.sess.run((maml.train_op, maml.merged))
        #
        #     if it % 20 == 0:
        #         maml.file_writer.add_summary(merged_summary, global_step=it)
        #         print(it)
        #     if it % 20000 == 0:
        #         maml.save_model(path=SAVING_PATH, step=it)
        #
        # if it != 0:
        #     maml.save_model(path=SAVING_PATH, step=it)

    else:
        maml.load_model('saved_models/omniglot/model-1000')
        print('Start testing the network')
        test_batch, test_batch_labels, test_val_batch, test_val_batch_labels = maml.sess.run(
            [maml.input_data, maml.input_labels, maml.input_validation, maml.input_validation_labels]
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

        outputs = maml.sess.run([maml.inner_model_out, maml], feed_dict={
            maml.input_data: test_val_batch,
            maml.input_labels: test_val_batch_labels,
        })

        print('model output:')
        outputs_np = np.argmax(outputs, axis=1)
        print(outputs_np)

        print('labels output:')
        labels_np = np.argmax(test_val_batch_labels.reshape(-1, 5), axis=1)
        print(labels_np)

        print('accuracy:')
        acc_num = np.sum(outputs_np == labels_np)
        acc = acc_num / 25.
        print(acc_num)
        print(acc)

    print('done')


if __name__ == '__main__':
    train_maml()
