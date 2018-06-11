import os
import random
import glob

import cv2
import tensorflow as tf
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


BASE_ADDRESS = '/home/siavash/UCF-101/'
UCF101_TFRECORDS = './ucf101_tfrecords/'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def one_hot_vector(labels, concept_size):
    return to_categorical(labels, concept_size)


class TraditionalDataset(object):
    def __init__(self, actions, class_sample_size, base_address):
        self.num_actions = len(actions)
        self.actions = actions
        self.action_samples = {}
        self.action_samples_train = {}
        self.action_samples_validation = {}

        for action_directory in self.actions:
            action_path = os.path.join(base_address, action_directory)
            self.action_samples[action_directory] = [
                os.path.join(action_path, action) for action in os.listdir(action_path)
            ]
            # Videos with smaller than 16 frames should not be used.
            should_be_removed_samples = []
            for sample in self.action_samples[action_directory]:
                if len(os.listdir(os.path.join(action_path, sample))) < 16:
                    print(sample)
                    should_be_removed_samples.append(sample)

            for sample in should_be_removed_samples:
                self.action_samples[action_directory].remove(sample)

        self.action_counter = 0
        self.within_class_counter = {action: 0 for action in self.actions}

        self.sample_k_samples(k=class_sample_size)
        self.shuffle_actions()
        # self.shuffle_within_actions()

    def sample_k_samples(self, k):
        for action in self.actions:
            self.shuffle_within_action(action)
            self.action_samples[action] = self.action_samples[action][:2 * k]
            self.action_samples_train[action] = self.action_samples[action][:k]
            self.action_samples_validation[action] = self.action_samples[action][k:]

    def shuffle_actions(self):
        random.shuffle(self.actions)

    def shuffle_within_action(self, action):
        random.shuffle(self.action_samples[action])

    def shuffle_within_actions(self):
        for action in self.actions:
            self.shuffle_within_action(action)

    def get_data_and_labels(self, files):
        """This function takes a list of list. Each list within the list corresponds to one class."""
        data = []
        labels = []
        label_counter = 0
        for label in files:
            for sample in label:
                frames_list = sorted(os.listdir(sample))
                start_frame = random.randint(0, len(frames_list) - 16)
                end_frame = start_frame + 16
                frames_list = frames_list[start_frame:end_frame]

                video_frames = [
                    cv2.resize(
                        np.array(plt.imread(os.path.join(sample, frame_name))), (112, 112)
                    )
                    for frame_name in frames_list
                ]

                video_frames = np.concatenate(video_frames).reshape(-1, 112, 112, 3)
                data.append(video_frames)
                labels.append(label_counter)
            label_counter += 1

        return np.concatenate(data).reshape(-1, 16, 112, 112, 3), \
            one_hot_vector(np.array(labels).reshape(-1, 1), concept_size=5)

    def next_batch(self, num_classes):
        action_begin = self.action_counter
        action_end = self.action_counter + num_classes
        if action_end <= self.num_actions:
            action_classes = self.actions[action_begin: action_end]
            self.action_counter += num_classes
        else:
            action_classes = self.actions[action_begin:]
            self.shuffle_actions()
            required_num_classes = num_classes - len(action_classes)
            action_classes.extend(self.actions[0: required_num_classes])
            self.action_counter += required_num_classes

        train_files = []
        validation_files = []
        for action in action_classes:
            train_samples = self.action_samples_train[action]
            validation_samples = self.action_samples_validation[action]

            train_files.append(train_samples)
            validation_files.append(validation_samples)

        return {
            'train': self.get_data_and_labels(train_files),
            'validation': self.get_data_and_labels(validation_files),
        }


class DataSetUtils(object):
    def create_tfrecord_dataset(self, base_address):
        labels = sorted(os.listdir(base_address))
        for label in labels:
            DataSetUtils.check_tf_directory(label)
            label_address = os.path.join(base_address, label)
            samples = sorted(os.listdir(label_address))
            for sample in samples:
                if sample.startswith('v_PommelHorse_g05'):
                    print('Ignoring {}'.format(sample))
                    continue

                sample_address = os.path.join(label_address, sample)
                frames_list = sorted(os.listdir(sample_address))
                if len(frames_list) < 16:
                    print('Ignoring {}'.format(sample))
                    continue

                video_frames = [
                    np.array(plt.imread(os.path.join(sample_address, frame_name))) for frame_name in frames_list
                ]
                video_frames = np.concatenate(video_frames).reshape(-1, 240, 320, 3)
                DataSetUtils.write_tf_record(video_frames, sample, action=label)

    @staticmethod
    def check_tf_directory(action):
        directory = os.path.join(UCF101_TFRECORDS, action)
        if not os.path.exists(directory):
            os.mkdir(directory)

    @staticmethod
    def write_tf_record(video_frames, sample_name, action):
        print(sample_name)
        tf_file_address = os.path.join(UCF101_TFRECORDS, action, sample_name) + '.tfrecord'
        if os.path.exists(tf_file_address):
            return

        writer = tf.python_io.TFRecordWriter(tf_file_address)
        feature = {
            'task': _bytes_feature(tf.compat.as_bytes(action)),
            'len': _int64_feature(video_frames.shape[0]),
            'video': _bytes_feature(tf.compat.as_bytes(video_frames.tostring()))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        writer.close()

    @staticmethod
    def concatenate_datasets(datasets):
        """ Concatenate a list of datasets together.
            Snippet by user2781994 (https://stackoverflow.com/a/49069420/624547)
        """
        ds0 = tf.data.Dataset.from_tensors(datasets[0])
        for ds1 in datasets[1:]:
            ds0 = ds0.concatenate(tf.data.Dataset.from_tensors(ds1))
        return ds0

    @staticmethod
    def get_datasets(directories, num_examples_per_class=5, num_classes_per_batch=5):
        def _parse(example_proto):
            features = {
                'task': tf.FixedLenFeature([], tf.string),
                'len': tf.FixedLenFeature([], tf.int64),
                'video': tf.FixedLenFeature([], tf.string),
            }
            parsed_example = tf.parse_single_example(example_proto, features)

            start_frame_number = tf.cond(
                tf.equal(parsed_example['len'], 16),
                lambda: tf.cast(0, tf.int64),
                lambda: tf.random_uniform([], minval=0, maxval=parsed_example['len'] - 16, dtype=tf.int64)
            )

            decoded_video = tf.decode_raw(parsed_example['video'], tf.uint8)
            reshaped_video = tf.reshape(decoded_video, shape=(-1, 240, 320, 3))
            resized_video = tf.cast(tf.image.resize_images(
                reshaped_video,
                size=(112, 112),
                method=tf.image.ResizeMethod.BILINEAR
            ), tf.uint8)

            clip = resized_video[start_frame_number:start_frame_number + 16, :, :, :]

            parsed_example['video'] = clip
            return parsed_example

        num_classes = len(directories)
        batch_size = num_classes_per_batch * num_examples_per_class
        per_class_datasets = [
            tf.data.TFRecordDataset(tf.data.Dataset.list_files(directory)).map(_parse) for directory in directories
        ]
        classes_per_batch_dataset = tf.contrib.data.Counter().map(
            lambda _: tf.random_shuffle(tf.range(num_classes))[:num_classes_per_batch]
        )

        class_dataset = classes_per_batch_dataset.flat_map(
            lambda classes: tf.data.Dataset.from_tensor_slices(
                tf.one_hot(classes, num_classes)).repeat(2 * num_examples_per_class)
        )

        dataset = tf.contrib.data.sample_from_datasets(per_class_datasets, class_dataset).batch(2 * batch_size)
        return dataset


def get_traditional_dataset(num_train_actions, base_address, train_actions=None, test_actions=None, class_sample_size=5):
    if train_actions is None:
        train_actions = sorted(os.listdir(base_address))[:num_train_actions]
    if test_actions is None:
        test_actions = sorted(os.listdir(base_address))[num_train_actions:]

    train_dataset = TraditionalDataset(train_actions, class_sample_size=class_sample_size, base_address=base_address)
    test_dataset = TraditionalDataset(test_actions, class_sample_size=class_sample_size, base_address=base_address)
    return train_dataset, test_dataset


def get_fast_dataset(directories):
    directories = sorted(directories)
    directories = [os.path.join(UCF101_TFRECORDS, directory) + '/*' for directory in directories]
    dataset = DataSetUtils.get_datasets(directories, num_classes_per_batch=5, num_examples_per_class=5)
    iterator = dataset.repeat(-1).make_one_shot_iterator()

    example = iterator.get_next()
    num_samples_per_batch = 5 * 5
    train_example = {
        'task': example['task'][:num_samples_per_batch],
        'video': example['video'][:num_samples_per_batch],
    }
    val_example = {
        'task': example['task'][num_samples_per_batch:],
        'video': example['video'][num_samples_per_batch:],
    }
    return train_example, val_example


def test_tfercord_dataset():
    directories = sorted(os.listdir(UCF101_TFRECORDS))
    train_example, val_example = get_fast_dataset(directories)

    tf.summary.FileWriter('logs/test/', tf.get_default_graph())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    from datetime import datetime
    begin = datetime.now()
    for i in range(10):
        tr_task, val_task, tr_vid, va_vid = sess.run(
            [train_example['task'], val_example['task'], train_example['video'], val_example['video']]
        )
        for i in range(25):
            print(tr_task[i])
            print(val_task[i])
            plt.subplot(211)
            plt.imshow(tr_vid[i, 0, :, :, :])
            plt.subplot(212)
            plt.imshow(va_vid[i, 0, :, :, :])
            plt.show()
    end = datetime.now()
    print(end - begin)


if __name__ == '__main__':
    # DataSetUtils().create_tfrecord_dataset()
    # test_traditional_dataset()
    test_tfercord_dataset()
