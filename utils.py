import random
import os

from scipy.misc import imresize
import matplotlib.pyplot as plt


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
