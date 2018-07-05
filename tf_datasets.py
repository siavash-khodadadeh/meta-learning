import os

import tensorflow as tf
import matplotlib.pyplot as plt


def parse_example(example_proto):
    features = {
        'task': tf.FixedLenFeature([], tf.string),
        'len': tf.FixedLenFeature([], tf.int64),
        'video': tf.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.parse_single_example(example_proto, features)
    return parsed_example


def extract_video(parsed_example):
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

    return clip


def get_ucf101_tf_dataset(dataset_address, num_classes, num_classes_per_batch, num_examples_per_class, one_hot=True):
    classes_list = sorted(os.listdir(dataset_address))
    mapping_strings = tf.constant(classes_list)
    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, num_oov_buckets=0, default_value=-1)

    def _parse_example(example):
        parsed_example = parse_example(example)
        feature = extract_video(parsed_example)

        example_address = parsed_example['task']
        label = tf.string_split([example_address], '/')
        label = label.values[0]
        label = table.lookup(label)
        if one_hot:
            label = tf.one_hot(label, depth=num_classes)
        return feature, label

    classes_list = [dataset_address + class_name + '/*' for class_name in classes_list][:num_classes]
    per_class_datasets = [
        tf.data.TFRecordDataset(tf.data.Dataset.list_files(directory).repeat(-1)) for directory in classes_list
    ]
    classes_per_batch_dataset = tf.contrib.data.Counter().map(
        lambda _: tf.random_shuffle(tf.range(num_classes))[:num_classes_per_batch]
    )
    class_dataset = classes_per_batch_dataset.flat_map(
        lambda classes: tf.data.Dataset.from_tensor_slices(
            tf.one_hot(classes, num_classes)
        ).repeat(num_examples_per_class)
    )
    dataset = tf.contrib.data.sample_from_datasets(per_class_datasets, class_dataset)
    dataset = dataset.map(_parse_example)

    meta_batch_size = num_classes_per_batch * num_examples_per_class
    dataset = dataset.batch(meta_batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator.get_next(), iterator


def test_get_ucf101_tf_dataset():
    actions = sorted(os.listdir('/home/siavash/programming/FewShotLearning/ucf101_tfrecords/'))
    next_example, iterator = get_ucf101_tf_dataset(
        '/home/siavash/programming/FewShotLearning/ucf101_tfrecords/',
        num_classes=101,
        num_classes_per_batch=20,
        num_examples_per_class=1,
        one_hot=False
    )
    input_ph = next_example[0]
    label = next_example[1]
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        tf.tables_initializer().run()
        for _ in range(150):
            data_np, label_np = sess.run((input_ph, label))
            print(label_np)
            print([actions[label] for label in label_np])
            for subplot_index in range(1, 21):
                plt.subplot(4, 5, subplot_index)
                plt.imshow(data_np[subplot_index - 1, 0, :, :, :])
            plt.show()


if __name__ == '__main__':
    test_get_ucf101_tf_dataset()
