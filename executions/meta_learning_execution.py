import os
import random

import tensorflow as tf


from tf_datasets import create_ucf101_data_feed_for_k_sample_per_action_iterative_dataset, create_data_feed_for_train
from models import ModelAgnosticMetaLearning, C3DNetwork
import settings


META_TRAIN = True  # true if we want to do meta train otherwise performing meta-test.
DATASET = 'kinetics'  # from 'kinetics', 'ucf-101', 'omniglot'.
N = 20  # Train an N-way classifier.
K = 1  # Train a K-shot learner

NUM_ITERATIONS = 10000
REPORT_AFTER_STEP = 20
SAVE_AFTER_STEP = 500
BATCH_SIZE = 10  # The batch size.
META_LEARNING_RATE = 0.00001
LEARNING_RATE = 0.001

NUM_META_TEST_ITERATIONS = 5
REPORT_AFTER_META_TEST_STEP = 1

NUM_GPUS = 2  # Number of GPUs to use for training.
RANDOM_SEED = 100  # Random seed value. Set it to -1 in order not to use a random seed.
STARTING_POINT_MODEL_ADDRESS = os.path.join(settings.PROJECT_ADDRESS, 'MAML/sports1m_pretrained.model')

META_TEST_STARTING_MODEL = settings.SAVED_MODELS_ADDRESS + '/backups/kinetics-from-server/-8000'


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

    gpu_devices = ['/gpu:{}'.format(gpu_id) for gpu_id in range(NUM_GPUS)]

    if DATASET == 'ucf-101':
        base_address = '/home/siavash/ucf101_tfrecords/'
        # '/home/siavash/programming/FewShotLearning/ucf101_tfrecords/'
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
        print(test_actions[:BATCH_SIZE * NUM_GPUS])
        input_data_ph, input_labels_ph, iterator = create_ucf101_data_feed_for_k_sample_per_action_iterative_dataset(
            dataset_address=base_address,
            k=K,
            batch_size=BATCH_SIZE * NUM_GPUS,
            actions_include=test_actions[:BATCH_SIZE * NUM_GPUS],
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
        gpu_devices=gpu_devices,
        meta_learn_rate=META_LEARNING_RATE,
        learning_rate=LEARNING_RATE,
        log_device_placement=False,
        num_classes=N
    )

    maml.sess.run(tf.tables_initializer())
    maml.sess.run(iterator.initializer)
    # data_np, labels_np = maml.sess.run((input_data_ph, input_labels_ph))
    # for i in range(N):
    #     print(labels_np[i, :])
    #     import matplotlib.pyplot as plt
    #     plt.imshow(data_np[i, 0, :, :, :])
    #     plt.show()

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
        maml.meta_test(NUM_META_TEST_ITERATIONS, save_model_per_x_iterations=REPORT_AFTER_META_TEST_STEP)
