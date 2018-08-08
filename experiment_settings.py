import os

import settings


META_TRAIN = False  # true if we want to do meta train otherwise performing meta-test.
DATASET = 'ucf-101'  # from 'kinetics', 'ucf-101', 'omniglot' or 'diva'.
N = 5  # Train an N-way classifier.
K = 1  # Train a K-shot learner

NUM_ITERATIONS = 100000
REPORT_AFTER_STEP = 20
SAVE_AFTER_STEP = 2000
BATCH_SIZE = 5  # The batch size.
META_LEARNING_RATE = 0.00001
LEARNING_RATE = 0.01

NUM_META_TEST_ITERATIONS = 301
SAVE_AFTER_META_TEST_STEP = 30

NUM_GPUS = 1  # Number of GPUs to use for training.
RANDOM_SEED = 100  # Random seed value. Set it to -1 in order not to use a random seed.
STARTING_POINT_MODEL_ADDRESS = os.path.join(settings.PROJECT_ADDRESS, 'MAML/sports1m_pretrained.model')


META_TEST_MODEL = 'kinetics/meta-train/5-way-classifier/1-shot/' \
                  'batch-size-5/num-gpus-1/random-seed-100/num-iterations-100000/meta-learning-rate-1e-05/' \
                  'learning-rate-0.01/-2000'

# META_TEST_MODEL = 'ucf-101/meta-train/5-way-classifier/1-shot/batch-size-5/num-gpus-1/random-seed-100/' \
#                   'num-iterations-1000/meta-learning-rate-1e-05/learning-rate-0.001/-1000'

# META_TEST_MODEL = 'backups/kinetics-from-server/logs/-10000'
META_TEST_STARTING_MODEL = os.path.join(settings.SAVED_MODELS_ADDRESS, META_TEST_MODEL)


test_actions = sorted(os.listdir(settings.UCF101_TF_RECORDS_ADDRESS))[-20:]

diva_test_actions = [
    ['activity_carrying', 'Closing', 'Interacts', 'specialized_talking_phone', 'vehicle_turning_left'],
    ['activity_sitting', 'vehicle_u_turn', 'Loading', 'Open_Trunk', 'Riding'],
    ['Closing_Trunk', 'Entering', 'Talking', 'specialized_texting_phone', 'vehicle_turning_right'],
    ['Exiting', 'Opening', 'Pull', 'Transport_HeavyCarry', 'Unloading'],
]