import os

import tensorflow as tf
import numpy as np

import settings
from models import ModelAgnosticMetaLearning, C3DNetwork

base_address = '/home/siavash/DIVA-TF-RECORDS/validation'
action_labels = {
    'close_trunk': 0,
    'specialized_talking_phone': 2,
    'specialized_texting_phone': 3,
    'vehicle_u_turn': 4,
    'open_trunk': 1,
}

with tf.variable_scope('train_data'):
    input_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
    input_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5])
    tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=5)

with tf.variable_scope('validation_data'):
    val_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
    val_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5])
    tf.summary.image('validation', val_data_ph[:, 0, :, :, :], max_outputs=5)


maml = ModelAgnosticMetaLearning(
    C3DNetwork,
    input_data_ph,
    input_labels_ph,
    val_data_ph,
    val_labels_ph,
    log_dir=None,
    saving_path=None,
    num_gpu_devices=1,
    meta_learn_rate=0.00001,
    learning_rate=0.001,
    log_device_placement=False,
    num_classes=5
)


maml.load_model(path=settings.SAVED_MODELS_ADDRESS + '/meta-test/model/-5')


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
        guessed_label = np.argmax(outputs)
        print(guessed_label)
        if guessed_label == label:
            correct += 1
        total += 1

    print('accuracy:')
    print(float(correct) / float(total))
