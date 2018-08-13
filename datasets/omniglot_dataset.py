import os

import tensorflow as tf
import matplotlib.pyplot as plt

from datasets.ucf101_data_generator import _bytes_feature, _int64_feature
import settings
from utils import prepare_classes_list_and_table


def combine_first_two_axes(data):
    data_current_shape = data.shape
    data_target_shape = [-1]

    for i in range(2, len(data_current_shape)):
        data_target_shape.append(data_current_shape[i])

    return tf.reshape(data, data_target_shape)


def create_tf_records():
    for alphabet_name in os.listdir(settings.OMNIGLOT_RAW_ADDRESS):
        alphabet_directory = os.path.join(settings.OMNIGLOT_RAW_ADDRESS, alphabet_name)

        for character in os.listdir(alphabet_directory):
            character_directory = os.path.join(alphabet_directory, character)
            tf_directory = os.path.join(settings.OMNIGLOT_TF_RECORD_ADDRESS, alphabet_name + '_' + character)
            if not os.path.exists(tf_directory):
                os.mkdir(tf_directory)

            for sample in os.listdir(character_directory):
                tf_file_address = os.path.join(tf_directory, sample.replace('.png', '.tfrecord'))
                sample_address = os.path.join(character_directory, sample)
                print(sample_address)
                image = plt.imread(sample_address)
                writer = tf.python_io.TFRecordWriter(tf_file_address)
                feature = {
                    'task': _bytes_feature(tf.compat.as_bytes(alphabet_name + '_' + character)),
                    'data': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                writer.close()


def per_directory_dataset(directory_glob):
    return tf.data.TFRecordDataset(tf.data.Dataset.list_files(directory_glob, shuffle=True))


def get_omniglot_tf_record_dataset(num_classes, num_samples_per_class, meta_batch_size, real_labels=False):
    characters, table = prepare_classes_list_and_table(
        settings.OMNIGLOT_TF_RECORD_ADDRESS,
        actions_include=None,
        actions_exclude=None
    )

    def _parse_example(example_proto):
        features = {
            'task': tf.FixedLenFeature([], tf.string),
            'data': tf.FixedLenFeature([], tf.string),
        }
        parsed_example = tf.parse_single_example(example_proto, features)
        decoded_data = tf.decode_raw(parsed_example['data'], tf.float32)
        decoded_data = tf.reshape(decoded_data, shape=(28, 28))
        return decoded_data, table.lookup(parsed_example['task'])

    characters = [
        os.path.join(settings.OMNIGLOT_TF_RECORD_ADDRESS, character) + '/*'
        for character in sorted(os.listdir(settings.OMNIGLOT_TF_RECORD_ADDRESS))
    ]

    datasets = tf.data.TFRecordDataset.from_tensor_slices(characters)
    datasets = datasets.shuffle(buffer_size=1000)

    dataset = datasets.interleave(
        per_directory_dataset, cycle_length=len(characters), block_length=num_samples_per_class * 2
    )

    dataset = dataset.take(len(characters) * num_samples_per_class * 2)

    dataset = dataset.repeat(-1)

    dataset = dataset.map(_parse_example)

    dataset = dataset.batch(num_classes * num_samples_per_class * 2)

    dataset = dataset.batch(meta_batch_size)

    iterator = dataset.make_initializable_iterator()

    data, classes = iterator.get_next()

    if not real_labels:
        classes = tf.reshape(tf.range(num_classes), (-1, 1))
        classes = tf.tile(classes, (1, num_samples_per_class * 2))
        classes = tf.reshape(classes, (1, -1))
        classes = tf.tile(classes, (meta_batch_size, 1))
        classes = tf.one_hot(classes, depth=num_classes)
    else:
        # pass
        classes = tf.one_hot(classes, depth=len(characters))

    train_data = combine_first_two_axes(data[:, ::2, ...])
    train_data = tf.reshape(train_data, (-1, 28, 28, 1))
    validation_data = combine_first_two_axes(data[:, 1::2, ...])
    validation_data = tf.reshape(validation_data, (-1, 28, 28, 1))

    train_classes = combine_first_two_axes(classes[:, ::2, ...])
    validation_classes = combine_first_two_axes(classes[:, 1::2, ...])

    with tf.variable_scope('train_data'):
        tf.summary.image('train', train_data, max_outputs=25)

    with tf.variable_scope('validation_data'):
        tf.summary.image('validation', validation_data, max_outputs=25)

    return train_data, train_classes, validation_data, validation_classes, iterator, table

if __name__ == '__main__':
    # create_tf_records()
    image_tr, label_tr, image_va, label_va, iter, tble = get_omniglot_tf_record_dataset(5, 5, 25, real_labels=False)
    with tf.Session() as sess:
        sess.run(iter.initializer)
        sess.run(tf.initialize_all_tables())
        for step in range(int(1624 / 5)):
            print(step)
            image_np_tr, label_np_tr, image_np_va, label_np_va = sess.run((image_tr, label_tr, image_va, label_va))
            print('\nbatch samples: \n')
            print('tr')
            print(image_np_tr.shape)
            print(label_np_tr)
            print('val')
            print(image_np_va.shape)
            print(label_np_va)

            # for batch_step in range(5 *5 ):
            #     plt.imshow(image_np_tr[batch_step, ...])
            #     plt.show()
            #
            #     plt.imshow(image_np_va[batch_step, ...])
            #     plt.show()
