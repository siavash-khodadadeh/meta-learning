import os
import pickle

import tensorflow as tf
import numpy as np

import settings
from models import ModelAgnosticMetaLearning, C3DNetwork

base_address = '/home/siavash/DIVA-TF-RECORDS/validation'
labels_base_address = '/home/siavash/DIVA-FewShot/validation'


REAL_LABELS = {
    5: "specialized_texting_phone",
    12: "specialized_talking_phone",
    22: "Open_Trunk",
    23: "Closing_Trunk",
    24: "vehicle_u_turn",
}


network_labels_real_labels_mapping = {
    0: 23,
    1: 22,
    2: 12,
    3: 5,
    4: 24,
}


action_labels = {
    'specialized_texting_phone': 16,
    'specialized_talking_phone': 15,
    'Unloading': 12,
    'Transport_HeavyCarry': 11,
    'Talking': 10,
    'activity_carrying': 13,
    'Closing': 0,
    'vehicle_u_turn': 19,
    'Closing_Trunk': 1,
    'vehicle_turning_right': 18,
    'Entering': 2,
    'Exiting': 3,
    'Open_Trunk': 6,
    'activity_sitting': 14,
    'Interacts': 4,
    'vehicle_turning_left': 17,
    'Loading': 5,
    'Pull': 8,
    'Opening': 7,
    'Riding': 9,
}

with tf.variable_scope('train_data'):
    input_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
    input_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, len(action_labels)])
    tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=len(action_labels))

with tf.variable_scope('validation_data'):
    val_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
    val_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, len(action_labels)])
    tf.summary.image('validation', val_data_ph[:, 0, :, :, :], max_outputs=len(action_labels))


maml = ModelAgnosticMetaLearning(
    C3DNetwork,
    input_data_ph,
    input_labels_ph,
    val_data_ph,
    val_labels_ph,
    log_dir=settings.BASE_LOG_ADDRESS + '/logs/diva/',
    saving_path=None,
    num_gpu_devices=1,
    meta_learn_rate=0.00001,
    learning_rate=0.001,
    log_device_placement=False,
    num_classes=len(action_labels)
)


maml.load_model(path=settings.SAVED_MODELS_ADDRESS + '/meta-test/model/-90')


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


for action, label in action_labels.items():
    correct = 0
    total = 0
    guess_table = [0] * len(action_labels)
    print(action)
    for file_address in os.listdir(os.path.join(base_address, action)):
        tf_record_address = os.path.join(base_address, action, file_address)
        dataset = tf.data.TFRecordDataset([tf_record_address])
        dataset = dataset.map(extract_video)
        iterator = dataset.make_one_shot_iterator()
        video = iterator.get_next()
        video_np = maml.sess.run(video).reshape(1, 16, 112, 112, 3)
        outputs = maml.sess.run(maml.inner_model_out, feed_dict={
            input_data_ph: video_np
        })

        # label_number_part = file_address[file_address.index('_') + 1:file_address.index('.')]
        # label_number_text = label_number_part[label_number_part.index('_') + 1:]
        # label_file = os.path.join(labels_base_address, action, 'labels', label_number_text + '_actions_39.pkl')
        # with open(label_file, 'rb') as f:
        #     data = pickle.load(f, encoding='latin1')
        #     labels_of_sample = np.where(data == 1)

        guessed_label = np.argmax(outputs)
        guess_table[guessed_label] += 1
        if guessed_label == label:
            correct += 1

        total += 1
    print('accuracy:')
    print(float(correct) / float(total))
    print('guess table:')
    print(guess_table)
