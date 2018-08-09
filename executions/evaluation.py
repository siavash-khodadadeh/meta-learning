import os

import tensorflow as tf
import numpy as np

from models import ModelAgnosticMetaLearning, C3DNetwork
from settings import UCF101_TF_RECORDS_ADDRESS
import settings


LOG_DIR = os.path.join(settings.PROJECT_ADDRESS, 'logs/temp/')

SAVED_MODEL_ADDRESS = os.path.join(settings.SAVED_MODELS_ADDRESS, 'meta-test/model/-300')


# TEST_ACTIONS = {
#     'YoYo': 19,
#     'WritingOnBoard': 18,
#     'WallPushups': 17,
#     'WalkingWithDog': 16,
#     'VolleyballSpiking': 15,
#     'ThrowDiscus': 11,
#     'TennisSwing': 10,
#     'Skijet': 0,
#     'SkyDiving': 1,
#     'SumoWrestling': 5,
#     'TaiChi': 9,
#     'SoccerJuggling': 2,
#     'SoccerPenalty': 3,
#     'UnevenBars': 14,
#     'TrampolineJumping': 12,
#     'StillRings': 4,
#     'Swing': 7,
#     'Typing': 13,
#     'TableTennisShot': 8,
#     'Surfing': 6,
# }


TEST_ACTIONS = {
    'StillRings': 4,
    'SoccerPenalty': 3,
    'SoccerJuggling': 2,
    'SkyDiving': 1,
    'Skijet': 0,
}


def extract_video(example):
    features = {
        'task': tf.FixedLenFeature([], tf.string),
        'len': tf.FixedLenFeature([], tf.int64),
        'video': tf.FixedLenFeature([], tf.string),
    }

    parsed_example = tf.parse_single_example(example, features)
    start_frame_number = tf.cond(
        tf.equal(parsed_example['len'], 16),
        lambda: tf.cast(0, tf.int64),
        lambda: tf.random_uniform([], minval=0, maxval=parsed_example['len'] - 16, dtype=tf.int64)
    )
    decoded_video = tf.decode_raw(parsed_example['video'], tf.uint8)
    resized_video = tf.reshape(decoded_video, shape=(-1, 112, 112, 3))

    clip = resized_video[start_frame_number:start_frame_number + 16, :, :, :]
    clip = tf.reshape(clip, (16, 112, 112, 3))

    return clip


def evaluate():
    with tf.variable_scope('train_data'):
        input_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        input_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, len(TEST_ACTIONS)])
        tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=25)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        val_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, len(TEST_ACTIONS)])
        tf.summary.image('validation', val_data_ph[:, 0, :, :, :], max_outputs=25)

    maml = ModelAgnosticMetaLearning(
        C3DNetwork,
        input_data_ph,
        input_labels_ph,
        val_data_ph,
        val_labels_ph,
        num_gpu_devices=1,
        log_dir=LOG_DIR,
        learning_rate=0.001,
        log_device_placement=False,
        saving_path=None,
        num_classes=len(TEST_ACTIONS),
    )

    maml.load_model(path=SAVED_MODEL_ADDRESS)

    correct = 0
    count = 0

    class_labels_couners = []

    for action in sorted(TEST_ACTIONS.keys()):
        class_label_counter = [0] * len(TEST_ACTIONS)
        print(action)
        for file_address in os.listdir(os.path.join(UCF101_TF_RECORDS_ADDRESS, action)):
            tf_record_address = os.path.join(UCF101_TF_RECORDS_ADDRESS, action, file_address)
            dataset = tf.data.TFRecordDataset([tf_record_address])
            dataset = dataset.map(extract_video)
            iterator = dataset.make_one_shot_iterator()
            video, labels = iterator.get_next()
            video_np = maml.sess.run((video, labels))
            video_np = video_np.reshape(1, 16, 112, 112, 3)

            outputs = maml.sess.run(maml.inner_model_out, feed_dict={
                maml.input_data: video_np,
            })

            label = np.argmax(outputs, 2)

            if label == TEST_ACTIONS[action]:
                correct += 1

            count += 1
            class_label_counter[label[0][0]] += 1

        print(class_label_counter)
        print(np.argmax(class_label_counter))
        class_labels_couners.append(class_label_counter)

    print('Accuracy: ')
    print(float(correct) / count)
    print(count)
    print(correct)

    confusion_matrix = np.array(class_labels_couners, dtype=np.float32).transpose()
    print('\n\n')
    print('confusion matrix')
    print(confusion_matrix)
    print('\n\n')
    columns_sum = np.sum(confusion_matrix, axis=0)
    rows_sum = np.sum(confusion_matrix, axis=1)

    counter = 0
    for action in sorted(TEST_ACTIONS.keys()):
        print(action)
        recall = confusion_matrix[counter][counter] / rows_sum[counter]
        precision = confusion_matrix[counter][counter] / columns_sum[counter]
        f1_score = 2 * precision * recall / (precision + recall)
        print('F1 Score: ')
        print(f1_score)
        counter += 1


if __name__ == '__main__':
    evaluate()
