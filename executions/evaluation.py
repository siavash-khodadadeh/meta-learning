import os

import tensorflow as tf
import numpy as np

from models import ModelAgnosticMetaLearning, C3DNetwork
from ucf101_data_generator import TraditionalDataset

LOG_DIR = 'logs/ucf101_transfer_learning/'
BASE_ADDRESS = '/home/siavash/UCF-101/'
# SAVED_MODEL_ADDRESS = 'saved_models/transfer_learning_80_5/model-400'
# SAVED_MODEL_ADDRESS = 'saved_models/transfer_learning_to_20_classes/model-400'
# SAVED_MODEL_ADDRESS = 'saved_models/transfer_learning_85/model-200'
# SAVED_MODEL_ADDRESS = 'saved_models/ucf101-fit/model-4'
# SAVED_MODEL_ADDRESS = 'saved_models/ucf101-fit/model-unsupervised-1'
# SAVED_MODEL_ADDRESS = 'saved_models/ucf101-fit/model-kinetics-trained-1'
SAVED_MODEL_ADDRESS = '/home/siavash/programming/FewShotLearning/saved_models/meta-test/model/-1'

# TEST_ACTIONS = {
#     'Surfing': 72,
#     'Typing': 79,
#     'Kayaking': 39,
#     'FieldHockeyPenalty': 24,
#     'BaseballPitch': 6,
# }

# TEST_ACTIONS = {
#     'Surfing': 0,
#     'Typing': 1,
#     'Kayaking': 2,
#     'FieldHockeyPenalty': 3,
#     'BaseballPitch': 4,
# }

# TEST_ACTIONS = {
#     'SoccerPenalty': 0,
#     'Kayaking': 1,
#     'Shotput': 2,
#     'Typing': 3,
#     'Bowling': 4,
#     'WritingOnBoard': 5,
#     'FrontCrawl': 6,
#     'JavelinThrow': 7,
#     'Surfing': 8,
#     'ShavingBeard': 9,
#     'CleanAndJerk': 10,
#     'BaseballPitch': 11,
#     'FrisbeeCatch': 12,
#     'SumoWrestling': 13,
#     'CuttingInKitchen': 14,
#     'PlayingSitar': 15,
#     'FloorGymnastics': 16,
#     'Fencing': 17,
#     'FieldHockeyPenalty': 18,
#     'MoppingFloor': 19,
# }

# TEST_ACTIONS = {
#     'PlayingSitar': 0,
#     'ShavingBeard': 1,
#     'CuttingInKitchen': 2,
#     'FloorGymnastics': 3,
#     'CleanAndJerk': 4,
#     'SumoWrestling': 5,
#     'Bowling': 6,
#     'Kayaking': 7,
#     'Shotput': 8,
#     'FrisbeeCatch': 9,
#     'Fencing': 10,
#     'MoppingFloor': 11,
#     'JumpingJack': 12,
#     'Surfing': 13,
#     'SoccerPenalty': 14,
#     'Typing': 15,
#     'FieldHockeyPenalty': 16,
#     'JavelinThrow': 17,
#     'FrontCrawl': 18,
#     'BaseballPitch': 19,
# }


# TEST_ACTIONS = {
#     'FrisbeeCatch': 0,
#     'ShavingBeard': 1,
#     'CliffDiving': 2,
#     'BandMarching': 3,
#     'FloorGymnastics': 4,
#     'Fencing': 5,
#     'JavelinThrow': 6,
#     'Basketball': 7,
#     'Bowling': 8,
#     'PlayingPiano': 9,
#     'FieldHockeyPenalty': 10,
#     'WritingOnBoard': 11,
#     'Archery': 12,
#     'Typing': 13,
#     'BabyCrawling': 14,
#     'ApplyEyeMakeup': 15,
#     'Biking': 16,
#     'BlowDryHair': 17,
#     'CuttingInKitchen': 18,
#     'Billiards': 19,
# }


TEST_ACTIONS = {
    'CleanAndJerk': 1,
    'MoppingFloor': 3,
    'FrontCrawl': 2,
    'Surfing': 4,
    'Bowling': 0,
}

# ALL UCF 101
# TEST_ACTIONS_LIST = sorted(os.listdir(BASE_ADDRESS))
# TEST_ACTIONS = {}

# counter = 0
# for action in TEST_ACTIONS_LIST:
#     TEST_ACTIONS[action] = counter
#     counter += 1
# ALL UCF 101 finished


def evaluate():
    with tf.variable_scope('train_data'):
        input_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        input_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=25)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        val_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        tf.summary.image('validation', val_data_ph[:, 0, :, :, :], max_outputs=25)

    maml = ModelAgnosticMetaLearning(
        C3DNetwork,
        input_data_ph,
        input_labels_ph,
        val_data_ph,
        val_labels_ph,
        log_dir=LOG_DIR,
        learning_rate=0.001,
        log_device_placement=False,
        saving_path=None,
        num_classes=5,
    )

    maml.load_model(path=SAVED_MODEL_ADDRESS)

    correct = 0
    count = 0
    for action in TEST_ACTIONS.keys():
        class_label_counter = [0] * len(TEST_ACTIONS)
        print(action)
        for file_address in os.listdir(BASE_ADDRESS + action):
            video_address = BASE_ADDRESS + action + '/' + file_address
            if len(os.listdir(video_address)) < 16:
                continue

            video, _ = TraditionalDataset.get_data_and_labels(None, [[video_address]], num_classes=len(TEST_ACTIONS))

            outputs = maml.sess.run(maml.inner_model_out, feed_dict={
                maml.input_data: video,
            })

            #  If doing Yogesh's suggestion
            # ind = np.argmax(
            #     (outputs[0][0, 72], outputs[0][0, 79], outputs[0][0, 39], outputs[0][0, 24], outputs[0][0, 6])
            # )
            # label = [72, 79, 39, 24, 6][ind]

            #  Otherwise
            label = np.argmax(outputs, 2)

            if label == TEST_ACTIONS[action]:
                correct += 1

            count += 1
            class_label_counter[label[0][0]] += 1

        print(class_label_counter)
        print(np.argmax(class_label_counter))

    print('Accuracy: ')
    print(float(correct) / count)
    print(count)
    print(correct)


if __name__ == '__main__':
    evaluate()
