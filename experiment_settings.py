import os

import settings


META_TRAIN = False  # true if we want to do meta train otherwise performing meta-test.
DATASET = 'ucf-101'  # from 'kinetics', 'ucf-101', 'omniglot' or 'diva'.
N = 80  # Train an N-way classifier.
K = 1  # Train a K-shot learner

NUM_ITERATIONS = 10000
REPORT_AFTER_STEP = 100
SAVE_AFTER_STEP = 2000
BATCH_SIZE = 40  # The batch size.
META_LEARNING_RATE = 0.001
LEARNING_RATE = 0.01
NUM_GPUS = 2  # Number of GPUs to use for training.
RANDOM_SEED = -1  # Random seed value. Set it to -1 in order not to use a random seed.
FIRST_OREDER_APPROXIMATION = True
BATCH_NORMALIZATION = True

META_TEST_STARTING_MODEL = os.path.join(
    'kinetics',
    'meta-train',
    'multiple-gpus',
    'with-first-order-approximation',
    'with-batch-normalization',
    '80-way-classifier',
    '1-shot',
    'batch-size-40',
    'num-gpus-2',
    'random-seed--1',
    'num-iterations-10000',
    'meta-learning-rate-0.001',
    'learning-rate-0.01',
    '4000'
)


NUM_META_TEST_ITERATIONS = 5
SAVE_AFTER_META_TEST_STEP = 1

STARTING_POINT_MODEL_ADDRESS = os.path.join(settings.PROJECT_ADDRESS, 'MAML/sports1m_pretrained.model')
# STARTING_POINT_MODEL_ADDRESS = None

test_actions = sorted(os.listdir(settings.UCF101_TF_RECORDS_ADDRESS))[-80:]

diva_test_actions = [
    ['activity_carrying', 'Closing', 'Interacts', 'specialized_talking_phone', 'vehicle_turning_left'],
    ['activity_sitting', 'vehicle_u_turn', 'Loading', 'Open_Trunk', 'Riding'],
    ['Closing_Trunk', 'Entering', 'Talking', 'specialized_texting_phone', 'vehicle_turning_right'],
    ['Exiting', 'Opening', 'Pull', 'Transport_HeavyCarry', 'Unloading'],
]
