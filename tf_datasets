import os

import tensorflow as tf


def get_ucf101_tf_dataset(dataset_address, num_classes, num_classes_per_batch, num_examples_per_class, one_hot=True):
    classes_list = os.listdir(dataset_address)
    mapping_strings = tf.constant(classes_list)
    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, num_oov_buckets=0, default_value=-1)

    def parse_example(example_address):
        feature = example_address + '/00001.jpg'
        feature = tf.expand_dims(feature, 0)
        filename_queue = tf.train.string_input_producer(feature)
        file_reader = tf.WholeFileReader()
        feature = file_reader.read(filename_queue)
        print(feature.shape)
        feature = tf.image.decode_jpeg(file_reader.read(feature))

        label = tf.string_split([example_address], '/')
        label = label.values[3]
        label = table.lookup(label)
        if one_hot:
            label = tf.one_hot(label, depth=num_classes)
        return feature, label

    classes_list = [dataset_address + class_name + '/*' for class_name in classes_list][:num_classes]
    per_class_datasets = [tf.data.Dataset.list_files(directory).repeat(-1) for directory in classes_list]
    classes_per_batch_dataset = tf.contrib.data.Counter().map(
        lambda _: tf.random_shuffle(tf.range(num_classes))[:num_classes_per_batch]
    )
    class_dataset = classes_per_batch_dataset.flat_map(
        lambda classes: tf.data.Dataset.from_tensor_slices(
            tf.one_hot(classes, num_classes)
        ).repeat(num_examples_per_class)
    )
    dataset = tf.contrib.data.sample_from_datasets(per_class_datasets, class_dataset)
    dataset = dataset.map(parse_example)

    meta_batch_size = num_classes_per_batch * num_examples_per_class
    dataset = dataset.batch(meta_batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator.get_next(), iterator


def test_get_ucf101_tf_dataset():
    data, iterator = get_ucf101_tf_dataset(
        '/home/siavash/UCF-101/',
        num_classes=101,
        num_classes_per_batch=20,
        num_examples_per_class=1,
        one_hot=False
    )
    input_ph = data[0]
    label = data[1]
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        tf.tables_initializer().run()
        for _ in range(150):
            data_np, label_np = sess.run((input_ph, label))
            print(data_np)
            print(label_np)


if __name__ == '__main__':
    test_get_ucf101_tf_dataset()
