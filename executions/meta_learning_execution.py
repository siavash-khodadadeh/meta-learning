import os
import random

import tensorflow as tf


from datasets.tf_datasets import create_ucf101_data_feed_for_k_sample_per_action_iterative_dataset, \
    create_data_feed_for_train, create_diva_data_feed_for_k_sample_per_action_iterative_dataset, \
    create_diva_data_feed_for_k_sample_per_action_iterative_dataset_unique_class_each_batch
from models import ModelAgnosticMetaLearning, C3DNetwork
import settings


META_TRAIN = False  # true if we want to do meta train otherwise performing meta-test.
DATASET = 'diva'  # from 'kinetics', 'ucf-101', 'omniglot' or 'diva'.
N = 5  # Train an N-way classifier.
K = 1  # Train a K-shot learner

NUM_ITERATIONS = 1000
REPORT_AFTER_STEP = 20
SAVE_AFTER_STEP = 500
BATCH_SIZE = 5  # The batch size.
META_LEARNING_RATE = 0.0001
LEARNING_RATE = 0.001

NUM_META_TEST_ITERATIONS = 101
SAVE_AFTER_META_TEST_STEP = 30

NUM_GPUS = 1  # Number of GPUs to use for training.
RANDOM_SEED = 100  # Random seed value. Set it to -1 in order not to use a random seed.
STARTING_POINT_MODEL_ADDRESS = os.path.join(settings.PROJECT_ADDRESS, 'MAML/sports1m_pretrained.model')


META_TEST_MODEL = '/backups/kinetics-from-server/-9900'
META_TEST_STARTING_MODEL = settings.SAVED_MODELS_ADDRESS + META_TEST_MODEL


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

    model_dir = os.path.join(
        DATASET,
        'meta-train',
        '{}-way-classifier'.format(N),
        '{}-shot'.format(K),
        'batch-size-{}'.format(BATCH_SIZE),
        'num-gpus-{}'.format(NUM_GPUS),
        'random-seed-{}'.format(RANDOM_SEED),
        'num-iterations-{}'.format(NUM_ITERATIONS),
        'meta-learning-rate-{}'.format(META_LEARNING_RATE),
        'learning-rate-{}'.format(LEARNING_RATE),
    )

    if META_TRAIN:
        log_dir = os.path.join(settings.BASE_LOG_ADDRESS, model_dir)
        saving_path = os.path.join(settings.SAVED_MODELS_ADDRESS, model_dir)
    else:
        log_dir = os.path.join(settings.BASE_LOG_ADDRESS, 'meta-test')
        saving_path = os.path.join(settings.SAVED_MODELS_ADDRESS, 'meta-test', 'model')

    if DATASET == 'ucf-101':
        base_address = '/home/siavash/ucf101_tfrecords/'
        # '/home/siavash/programming/FewShotLearning/ucf101_tfrecords/'
    elif DATASET == 'diva':
        base_address = '/home/siavash/DIVA-TF-RECORDS/train/'
    else:
        base_address = '/home/siavash/kinetics_tfrecords/'

    if META_TRAIN:
        input_data_ph, input_labels_ph, val_data_ph, val_labels_ph, iterator = create_data_feed_for_train(
            base_address=base_address,
            test_actions=test_actions,
            batch_size=BATCH_SIZE * NUM_GPUS,
            k=K,
            n=N,
            random_labels=False
        )
    else:
        if DATASET == 'ucf-101' or DATASET == 'kinetics':
            print(test_actions[:BATCH_SIZE * NUM_GPUS])
            input_data_ph, input_labels_ph, iterator, table = \
                create_ucf101_data_feed_for_k_sample_per_action_iterative_dataset(
                    dataset_address=base_address,
                    k=K,
                    batch_size=BATCH_SIZE * NUM_GPUS,
                    actions_include=test_actions[:BATCH_SIZE * NUM_GPUS],
                )
            val_data_ph = input_data_ph
            val_labels_ph = input_labels_ph
        else:
            # input_data_ph, input_labels_ph, iterator = create_diva_data_feed_for_k_sample_per_action_iterative_dataset(
            #     dataset_address=base_address,
            #     k=K,
            #     batch_size=BATCH_SIZE * NUM_GPUS,
            # )
            input_data_ph, input_labels_ph, iterator, table = \
                create_diva_data_feed_for_k_sample_per_action_iterative_dataset_unique_class_each_batch(
                    dataset_address=base_address
                )

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
        num_gpu_devices=NUM_GPUS,
        meta_learn_rate=META_LEARNING_RATE,
        learning_rate=LEARNING_RATE,
        log_device_placement=False,
        num_classes=N
    )

    maml.sess.run(tf.tables_initializer())
    maml.sess.run(iterator.initializer)
    if not META_TRAIN:
        print(maml.sess.run(table.export()))

    return maml


if __name__ == '__main__':
    maml = initialize()
    if META_TRAIN:
        maml.load_model(path=STARTING_POINT_MODEL_ADDRESS, load_last_layer=False)
        maml.meta_train(
            num_iterations=NUM_ITERATIONS + 1,
            report_after_x_step=REPORT_AFTER_STEP,
            save_after_x_step=SAVE_AFTER_STEP
        )
    else:
        maml.load_model(META_TEST_STARTING_MODEL)
        maml.meta_test(NUM_META_TEST_ITERATIONS, save_model_per_x_iterations=SAVE_AFTER_META_TEST_STEP)
