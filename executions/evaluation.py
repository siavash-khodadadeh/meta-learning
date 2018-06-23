import os

import tensorflow as tf
import numpy as np

from models import ModelAgnosticMetaLearning, C3DNetwork
from ucf101_data_generator import TraditionalDataset

LOG_DIR = 'logs/ucf101_transfer_learning/'
BASE_ADDRESS = '/home/siavash/UCF-101/'
# SAVED_MODEL_ADDRESS = 'saved_models/transfer_learning_80_5/model-400'
# SAVED_MODEL_ADDRESS = 'saved_models/transfer_learning_85/model-200'
# SAVED_MODEL_ADDRESS = 'saved_models/ucf101-fit/model-4'
SAVED_MODEL_ADDRESS = 'saved_models/ucf101-fit/model-unsupervised-4'

TEST_ACTIONS = {
    'Surfing': 0,
    'Typing': 1,
    'Kayaking': 2,
    'FieldHockeyPenalty': 3,
    'BaseballPitch': 4,
}

# TEST_ACTIONS = {
#     'Surfing': 72,
#     'Typing': 79,
#     'Kayaking': 39,
#     'FieldHockeyPenalty': 24,
#     'BaseballPitch': 6,
# }


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
        train=False,
        log_device_placement=False
    )

    maml.load_model(path=SAVED_MODEL_ADDRESS)

    correct = 0
    count = 0
    for action in TEST_ACTIONS.keys():
        for file_address in os.listdir(BASE_ADDRESS + action):
            video_address = BASE_ADDRESS + action + '/' + file_address
            if len(os.listdir(video_address)) < 16:
                continue

            video, _ = TraditionalDataset.get_data_and_labels(None, [[video_address]], num_classes=5)

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

    print('Accuracy: ')
    print(float(correct) / count)
    print(count)
    print(correct)


if __name__ == '__main__':
    evaluate()
