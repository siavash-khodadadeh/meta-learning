import os
import random

import tensorflow as tf


from tf_datasets import get_action_tf_dataset
from models import ModelAgnosticMetaLearning, C3DNetwork
import settings


META_TRAIN = True  # true if we want to do meta train otherwise performing meta-test.
DATASET = 'ucf-101'  # from 'kinetics', 'ucf-101', 'omniglot'.
N = 5  # Train an N-way classifier.
K = 1  # Train a k-shot learner

BATCH_SIZE = 5  # The batch size.
NUM_GPUS = 1  # Number of GPUs to use for training.
RANDOM_SEED = 100  # Random seed value. Set it to -1 in order not to use a random seed.
STARTING_POINT_MODEL_ADDRESS = os.path.join(settings.PROJECT_ADDRESS, 'MAML/sports1m_pretrained.model')

NUM_ITERATIONS = 1000
REPORT_AFTER_STEP = 20
SAVE_AFTER_STEP = 100


test_actions = [
    'CleanAndJerk',
    'MoppingFloor',
    'FrontCrawl',
    'Surfing',
    'Bowling',
    'SoccerPenalty',
    'SumoWrestling',
    'Shotput',
    'PlayingSitar',
    'FloorGymnastics',
    'Typing',
    'JumpingJack',
    'ShavingBeard',
    'FrisbeeCatch',
    'WritingOnBoard',
    'JavelinThrow',
    'Fencing',
    'FieldHockeyPenalty',
    'BaseballPitch',
    'CuttingInKitchen',
    'Kayaking',
]


def convert_to_fake_labels(labels):
    return tf.one_hot(tf.nn.top_k(labels, k=N).indices, depth=N)


def create_data_feed_for_ucf101(real_labels=False):
    with tf.variable_scope('dataset'):
        actions_exclude = test_actions if META_TRAIN else None
        actions_include = test_actions if not META_TRAIN else None

        dataset = get_action_tf_dataset(
            '/home/siavash/programming/FewShotLearning/ucf101_tfrecords/',
            num_classes=N,
            num_classes_per_batch=BATCH_SIZE,
            num_examples_per_class=K,
            one_hot=real_labels,
            actions_exclude=actions_exclude,
            actions_include=actions_include
        )

        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

    with tf.variable_scope('train_data'):
        input_data_ph = tf.cast(next_batch[0][:K * BATCH_SIZE], tf.float32)
        input_labels_ph = next_batch[1][:K * BATCH_SIZE]
        tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=K * BATCH_SIZE)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.cast(next_batch[0][K * BATCH_SIZE:], tf.float32)
        val_labels_ph = next_batch[1][K * BATCH_SIZE:]
        tf.summary.image('validation', val_data_ph[:, 0, :, :, :], max_outputs=K * BATCH_SIZE)

    if not real_labels:
        input_labels_ph = convert_to_fake_labels(input_labels_ph)
        val_labels_ph = convert_to_fake_labels(val_labels_ph)

    return input_data_ph, input_labels_ph, val_data_ph, val_labels_ph, iterator


def initialize():
    if RANDOM_SEED != -1:
        random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

    log_dir = os.path.join(
        settings.BASE_LOG_ADDRESS,
        DATASET, 'meta-train' if META_TRAIN else 'meta-test',
        '{}-way-classifier'.format(N),
        '{}-shot'.format(K),
        'batch-size-{}'.format(BATCH_SIZE),
        'num-gpus-{}'.format(NUM_GPUS),
        'random-seed-{}'.format(RANDOM_SEED),
        'num-iterations-{}'.format(NUM_ITERATIONS),
    )

    saving_path = os.path.join(
        settings.SAVED_MODELS_ADDRESS,
        DATASET, 'meta-train' if META_TRAIN else 'meta-test',
        '{}-way-classifier'.format(N),
        '{}-shot'.format(K),
        'batch-size-{}'.format(BATCH_SIZE),
        'num-gpus-{}'.format(NUM_GPUS),
        'random-seed-{}'.format(RANDOM_SEED),
        'num-iterations-{}'.format(NUM_ITERATIONS),
    )
    gpu_devices = ['/gpu:{}'.format(gpu_id) for gpu_id in range(NUM_GPUS)]

    # if DATASET == 'ucf101'
    input_data_ph, input_labels_ph, val_data_ph, val_labels_ph, iterator = create_data_feed_for_ucf101()

    maml = ModelAgnosticMetaLearning(
        C3DNetwork,
        input_data_ph,
        input_labels_ph,
        val_data_ph,
        val_labels_ph,
        log_dir=log_dir,
        saving_path=saving_path,
        gpu_devices=gpu_devices,
        meta_learn_rate=0.0001,
        learning_rate=0.001,
        log_device_placement=False,
        num_classes=N
    )

    maml.sess.run(tf.tables_initializer())
    maml.sess.run(iterator.initializer)
    return maml


if __name__ == '__main__':
    maml = initialize()
    if META_TRAIN:
        maml.load_model(path=STARTING_POINT_MODEL_ADDRESS, load_last_layer=False)
        maml.meta_train(
            num_iterations=NUM_ITERATIONS,
            report_after_step=REPORT_AFTER_STEP,
            save_after_step=SAVE_AFTER_STEP
        )
    else:
        maml.load_model(maml.saving_path + '-1000')
        data, labels = maml.sess.run((maml.input_data, maml.input_labels))
        maml.meta_test(data, labels, 5)
