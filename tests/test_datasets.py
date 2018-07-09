import os

import tensorflow as tf
import matplotlib.pyplot as plt

from tf_datasets import get_action_tf_dataset


def test_get_ucf101_tf_dataset():
    actions = sorted(os.listdir('/home/siavash/programming/FewShotLearning/ucf101_tfrecords/'))
    dataset, classes_list = get_action_tf_dataset(
        '/home/siavash/programming/FewShotLearning/ucf101_tfrecords/',
        num_classes_per_batch=20,
        num_examples_per_class=1,
        one_hot=False,
        actions_exclude=['ApplyEyeMakeup', 'Bowling']
    )

    iterator = dataset.make_initializable_iterator()

    next_example = iterator.get_next()
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
