import random
import os

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
