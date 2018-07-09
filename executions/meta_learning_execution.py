import os
import random

import tensorflow as tf


from tf_datasets import create_data_feed_for_ucf101, create_ucf101_data_feed_for_k_sample_per_action_iterative_dataset
from models import ModelAgnosticMetaLearning, C3DNetwork
import settings


META_TRAIN = False  # true if we want to do meta train otherwise performing meta-test.
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

META_TEST_STARTING_MODEL = settings.SAVED_MODELS_ADDRESS + '/ucf-101/meta-train/5-way-classifier/1-shot/batch-size-5/' \
                                                  'num-gpus-1/random-seed-100/num-iterations-1000/-900'


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

    if DATASET == 'ucf101' and META_TRAIN:
        input_data_ph, input_labels_ph, val_data_ph, val_labels_ph, iterator, real_labels, classes_list = \
            create_data_feed_for_ucf101(test_actions, META_TRAIN, BATCH_SIZE, K, N)
    else:
        input_data_ph, input_labels_ph, iterator = \
            create_ucf101_data_feed_for_k_sample_per_action_iterative_dataset(
                k=K,
                batch_size=BATCH_SIZE,
            )

        # val_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        # val_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, N])
        val_data_ph = input_data_ph
        val_labels_ph = input_labels_ph

    maml = ModelAgnosticMetaLearning(
        C3DNetwork,
        input_data_ph,
        input_labels_ph,
        val_data_ph,
        val_labels_ph,
        log_dir=log_dir,
        saving_path=saving_path,
        gpu_devices=gpu_devices,
        meta_learn_rate=0.00001,
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
            num_iterations=NUM_ITERATIONS + 1,
            report_after_step=REPORT_AFTER_STEP,
            save_after_step=SAVE_AFTER_STEP
        )
    else:
        maml.load_model(META_TEST_STARTING_MODEL)
        maml.meta_test(100)
