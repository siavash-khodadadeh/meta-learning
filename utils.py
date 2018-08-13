import random
import os

import tensorflow as tf
from scipy.misc import imresize
import matplotlib.pyplot as plt
from keras.utils import to_categorical


def pre_process(image):
    new_image = imresize(image, size=(28, 28)).reshape(28, 28, 1) / 255.
    new_image = (new_image - 0.5) * 2
    return new_image


def load_image(path):
    image = plt.imread(path)
    return pre_process(image)


def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x

    images = [
        (i, os.path.join(path, image))
        for i, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
        ]

    if shuffle:
        random.shuffle(images)
    return images


def one_hot_vector(labels, concept_size):
    return to_categorical(labels, concept_size)


def get_next_train_val_batch(train_dataset, validation_dataset, concept_size=10):
    num = train_dataset.num_shot_per_concept
    train_batch_data, train_batch_labels = train_dataset.next_batch(concept_size=concept_size)
    val_batch_data, val_batch_labels = validation_dataset.next_batch(concepts=train_batch_labels[0::num].reshape(-1))

    for label in range(concept_size):
        train_batch_labels[train_batch_labels == train_batch_labels[num * label]] = label
        val_batch_labels[val_batch_labels == val_batch_labels[num * label]] = label

    train_batch_labels = one_hot_vector(train_batch_labels, concept_size)
    val_batch_labels = one_hot_vector(val_batch_labels, concept_size)
    return train_batch_data, train_batch_labels, val_batch_data, val_batch_labels


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
     Note that this function provides a synchronization point across all towers.
     Args:
       tower_grads: List of lists of (gradient, variable) tuples. The outer list
         is over individual gradients. The inner list is over the gradient
         calculation for each tower.
     Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
     """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def prepare_classes_list_and_table(dataset_address, actions_include=None, actions_exclude=None):
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
