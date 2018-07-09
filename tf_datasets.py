import os

import tensorflow as tf


def parse_example(example_proto):
    features = {
        'task': tf.FixedLenFeature([], tf.string),
        'len': tf.FixedLenFeature([], tf.int64),
        'video': tf.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.parse_single_example(example_proto, features)
    return parsed_example


def convert_to_fake_labels(labels, num_classes):
    return tf.one_hot(tf.nn.top_k(labels, k=num_classes).indices, depth=num_classes)


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
    clip = tf.reshape(clip, (16, 112, 112, 3))

    return clip


def prepare_classes_list_and_table(dataset_address, actions_include, actions_exclude):
    classes_list = sorted(os.listdir(dataset_address))
    should_be_removed_actions = []

    if actions_include is not None:
        for action in classes_list:
            if action not in actions_include:
                should_be_removed_actions.append(action)

    if actions_exclude is not None:
        for action in actions_exclude:
            should_be_removed_actions.append(action)

    for action in should_be_removed_actions:
        if action in classes_list:
            classes_list.remove(action)

    mapping_strings = tf.constant(classes_list)
    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, num_oov_buckets=0, default_value=-1)
    return classes_list, table


def get_action_tf_dataset(
    dataset_address,
    num_classes_per_batch,
    num_examples_per_class,
    one_hot=True,
    actions_exclude=None,
    actions_include=None
):
    classes_list, table = prepare_classes_list_and_table(dataset_address, actions_include, actions_exclude)

    def _parse_example(example):
        parsed_example = parse_example(example)
        feature = extract_video(parsed_example)

        example_address = parsed_example['task']
        label = tf.string_split([example_address], '/')
        label = label.values[0]
        label = table.lookup(label)
        if one_hot:
            label = tf.one_hot(label, depth=len(classes_list))
        return feature, label

    classes_list = [dataset_address + class_name + '/*' for class_name in classes_list]
    per_class_datasets = [
        tf.data.TFRecordDataset(tf.data.Dataset.list_files(directory).repeat(-1)) for directory in classes_list
    ]
    classes_per_batch_dataset = tf.contrib.data.Counter().map(
        lambda _: tf.random_shuffle(tf.range(len(classes_list)))[:num_classes_per_batch]
    )
    class_dataset = classes_per_batch_dataset.flat_map(
        lambda classes: tf.data.Dataset.from_tensor_slices(
            tf.one_hot(classes, len(classes_list))
        ).repeat(2 * num_examples_per_class)
    )
    dataset = tf.contrib.data.sample_from_datasets(per_class_datasets, class_dataset)
    dataset = dataset.map(_parse_example)

    meta_batch_size = num_classes_per_batch * num_examples_per_class
    dataset = dataset.batch(2 * meta_batch_size)
    return dataset, classes_list


def create_data_feed_for_ucf101(test_actions, train, batch_size, k, n, real_labels=False):
    """Meta learning dataset for ucf101."""
    with tf.variable_scope('dataset'):
        actions_exclude = test_actions if train else None
        actions_include = test_actions if not train else None

        dataset, classes_list = get_action_tf_dataset(
            '/home/siavash/programming/FewShotLearning/ucf101_tfrecords/',
            num_classes_per_batch=batch_size,
            num_examples_per_class=k,
            one_hot=real_labels,
            actions_exclude=actions_exclude,
            actions_include=actions_include
        )

        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

    with tf.variable_scope('train_data'):
        input_data_ph = tf.cast(next_batch[0][:k * batch_size], tf.float32)
        input_labels_ph = next_batch[1][:k * batch_size]
        tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=k * batch_size)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.cast(next_batch[0][k * batch_size:], tf.float32)
        val_labels_ph = next_batch[1][k * batch_size:]
        tf.summary.image('validation', val_data_ph[:, 0, :, :, :], max_outputs=k * batch_size)

    real_input_labels = input_labels_ph
    if not real_labels:
        input_labels_ph = convert_to_fake_labels(input_labels_ph, n)
        val_labels_ph = convert_to_fake_labels(val_labels_ph, n)

    return input_data_ph, input_labels_ph, val_data_ph, val_labels_ph, iterator, real_input_labels, classes_list


def create_k_sample_per_action_iterative_dataset(
        dataset_address,
        k,
        batch_size,
        one_hot=True,
        actions_include=None,
        actions_exclude=None,
):
    classes_list, table = prepare_classes_list_and_table(dataset_address, actions_include, actions_exclude)

    def _parse_example(example):
        parsed_example = parse_example(example)
        feature = extract_video(parsed_example)

        example_address = parsed_example['task']
        label = tf.string_split([example_address], '/')
        label = label.values[0]
        label = table.lookup(label)
        if one_hot:
            label = tf.one_hot(label, depth=len(classes_list))

        return feature, label

    examples = list()
    for class_directory in classes_list:
        video_addresses = os.listdir(os.path.join(dataset_address, class_directory))[:k]
        for video_address in video_addresses:
            examples.append(os.path.join(dataset_address, class_directory, video_address))

    dataset = tf.data.TFRecordDataset(examples).shuffle(100).repeat(-1)
    dataset = dataset.map(_parse_example)
    dataset = dataset.batch(batch_size)
    return dataset
