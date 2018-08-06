import os
import sys

import tensorflow as tf
import numpy as np

import settings
from models import ModelAgnosticMetaLearning, C3DNetwork


def extract_video(example):
    features = {
        'task': tf.FixedLenFeature([], tf.string),
        'len': tf.FixedLenFeature([], tf.int64),
        'video': tf.FixedLenFeature([], tf.string),
        'labels': tf.FixedLenFeature([], tf.string),
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

    labels = tf.decode_raw(parsed_example['labels'], tf.uint8)
    labels = tf.cast(labels, tf.float32)

    return clip, labels


base_address = settings.DIVA_VALIDATION_TF_RECORDS_ADDRESS
labels_base_address = os.path.join(settings.DIVA_RAW_ADDRESS, 'validation')


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
    'Closing': 0,
    'Closing_Trunk': 1,
    'Entering': 2,
    'Exiting': 3,
    'Interacts': 4,
    'Loading': 5,
    'Open_Trunk': 6,
    'Opening': 7,
    'Pull': 8,
    'Riding': 9,
    'Talking': 10,
    'Transport_HeavyCarry': 11,
    'Unloading': 12,
    'activity_carrying': 13,
    'activity_sitting': 14,
    'specialized_talking_phone': 15,
    'specialized_texting_phone': 16,
    'vehicle_turning_left': 17,
    'vehicle_turning_right': 18,
    'vehicle_u_turn': 19,
}


hierarchy = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 0,
    6: 0,
    7: 0,
    8: 2,
    9: 3,
    10: 1,
    11: 2,
    12: 0,
    13: 2,
    14: 0,
    15: 4,
    16: 4,
    17: 5,
    18: 5,
    19: 5,
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


maml.load_model(path=settings.SAVED_MODELS_ADDRESS + '/meta-test/model/-300')


class_labels_counters = []
hierarchy_counters = []

for action in sorted(action_labels.keys()):
    correct = 0
    total = 0
    guess_table = [0] * len(action_labels)
    guess_hierarchy_table = [0] * len(hierarchy)
    print(action)
    sys.stdout.flush()
    for file_address in os.listdir(os.path.join(base_address, action))[:50]:
        tf_record_address = os.path.join(base_address, action, file_address)
        dataset = tf.data.TFRecordDataset([tf_record_address])
        dataset = dataset.map(extract_video)
        iterator = dataset.make_one_shot_iterator()
        video, labels = iterator.get_next()
        video_np, labels_np = maml.sess.run((video, labels))
        video_np = video_np.reshape(1, 16, 112, 112, 3)
        labels_np = labels_np.reshape(1, -1)

        outputs = maml.sess.run(maml.inner_model_out, feed_dict={
            input_data_ph: video_np
        })
        guessed_label = np.argmax(outputs)

        outputs = np.array(outputs).reshape(1, -1)
        outputs = 1 / (1 + np.exp(-outputs))
        print('network outputs: ')
        print(np.where(outputs > 0.2)[1])
        print('real labels: ')
        print(np.where(labels_np == 1)[1])

        guess_table[guessed_label] += 1
        guess_hierarchy_table[hierarchy[guessed_label]] += 1
        if hierarchy[guessed_label] == hierarchy[action_labels[action]]:
            correct += 1

        total += 1

    class_labels_counters.append(guess_table)
    hierarchy_counters.append(guess_hierarchy_table)
    print('accuracy:')
    print(float(correct) / float(total))
    print('guess table:')
    print(guess_table)

    sys.stdout.flush()


confusion_matrix = np.array(class_labels_counters, dtype=np.float32).transpose()
columns_sum = np.sum(confusion_matrix, axis=0)
rows_sum = np.sum(confusion_matrix, axis=1)

print('confusion matrix')
print(confusion_matrix)

hierarchy_confusion_matrix = np.array(hierarchy_counters, dtype=np.float32).transpose()
print('hierarchy confusion matrix')
print(hierarchy_confusion_matrix)

counter = 0
for action in sorted(action_labels.keys()):
    print(action)
    recall = confusion_matrix[counter][counter] / rows_sum[counter]
    precision = confusion_matrix[counter][counter] / columns_sum[counter]
    f1_score = 2 * precision * recall / (precision + recall)
    print('F1 Score: ')
    print(f1_score)
    counter += 1
    sys.stdout.flush()
