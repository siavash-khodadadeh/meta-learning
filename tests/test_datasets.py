import os

import tensorflow as tf
import matplotlib.pyplot as plt

from datasets.tf_datasets import create_k_sample_per_action_iterative_dataset, create_data_feed_for_train, \
    create_diva_data_feed_for_k_sample_per_action_iterative_dataset


def test_create_data_feed_for_ucf101():
    input_data_ph, input_labels_ph, val_data_ph, val_labels_ph, iterator = create_data_feed_for_train(
        base_address='/home/siavash/programming/FewShotLearning/ucf101_tfrecords/',
        test_actions=None,
        batch_size=15,
        k=1,
        n=101,
        random_labels=True
    )

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        tf.tables_initializer().run()
        for _ in range(150):
            data_np, labels_np, val_data_np, val_labels_np = sess.run(
                (input_data_ph, input_labels_ph, val_data_ph, val_labels_ph)
            )

            print(labels_np)
            print(val_labels_np)
            for vid in range(15):
                plt.imshow(data_np[vid, 0, :, :, :])
                plt.show()
                plt.imshow(val_data_np[vid, 0, :, :, :])
                plt.show()


def test_create_feed_for_diva():
    input_data_ph, input_labels_ph, iterator = create_diva_data_feed_for_k_sample_per_action_iterative_dataset(
        dataset_address='/home/siavash/DIVA-TF-RECORDS/train/',
        k=1,
        batch_size=5,
    )

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        tf.tables_initializer().run()
        for _ in range(150):
            data_np, labels_np = sess.run((input_data_ph, input_labels_ph))

            print(labels_np)
            for vid in range(15):
                plt.imshow(data_np[vid, 0, :, :, :])
                plt.show()


def test_create_k_sample_per_action_iterative_dataset():
    actions = sorted(os.listdir('/home/siavash/programming/FewShotLearning/ucf101_tfrecords/'))
    dataset = create_k_sample_per_action_iterative_dataset(
        '/home/siavash/programming/FewShotLearning/ucf101_tfrecords/',
        k=1,
        batch_size=15,
        one_hot=False,
    )

    iterator = dataset.make_initializable_iterator()
    next_example = iterator.get_next()
    input_ph = next_example[0]
    label = next_example[1]

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        tf.tables_initializer().run()
        for _ in range(200):
            vid, label_np = sess.run((input_ph, label))
            print(label_np)
            print([actions[label] for label in label_np])
            for subplot_index in range(1, 15):
                plt.subplot(3, 5, subplot_index)
                plt.imshow(vid[subplot_index - 1, 0, :, :, :])
            plt.show()


if __name__ == '__main__':
    # test_create_k_sample_per_action_iterative_dataset()
    # test_create_data_feed_for_ucf101()
    test_create_feed_for_diva()
