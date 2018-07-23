""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

import settings
from utils import get_images


class DataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, num_classes=5, num_train=1200, test_set=False):
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.img_size = (28, 28)
        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes
        # data that is pre-resized using PIL with lanczos filter
        data_folder = settings.PROJECT_ADDRESS + '/data/omniglot_resized/'

        character_folders = [
            os.path.join(data_folder, family, character)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for character in os.listdir(os.path.join(data_folder, family))
        ]

        # random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train -= num_val

        self.metatrain_character_folders = character_folders[:num_train]
        if test_set:
            self.metaval_character_folders = character_folders[num_train+num_val:]
        else:
            self.metaval_character_folders = character_folders[num_train:num_train+num_val]
        self.rotations = [0, 90, 180, 270]
        self.do_rotation = True

    def make_data_tensor(self, train=True, binary_classification=False):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        print('Generating filenames')
        all_filenames = []
        for _ in range(num_total_batches):
            sampled_character_folders = random.sample(folders, self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(
                sampled_character_folders,
                range(self.num_classes),
                nb_samples=self.num_samples_per_class,
                shuffle=False
            )
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            if binary_classification:
                labels = [1 if labels[i] == 3 else 0 for i in range(len(labels))]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)

        image = tf.image.decode_png(image_file)
        image.set_shape((self.img_size[0], self.img_size[1], 1))
        image = tf.reshape(image, [self.dim_input])
        image = tf.cast(image, tf.float32) / 255.0
        image = 1.0 - image  # invert

        num_preprocess_threads = 1  # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
            [image],
            batch_size=batch_image_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_image_size,
        )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]

            if self.do_rotation:
                # omniglot augments the dataset by rotating digits to create new classes
                # get rotation per class (e.g. 0,1,2,0,0 if there are 5 classes)
                rotations = tf.multinomial(tf.log([[1., 1.,1.,1.]]), self.num_classes)

            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs * self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch, true_idxs))
                if self.do_rotation:
                    new_list[-1] = tf.stack([tf.reshape(tf.image.rot90(
                        tf.reshape(new_list[-1][ind], [self.img_size[0], self.img_size[1], 1]),
                        k=tf.cast(rotations[0, class_idxs[ind]], tf.int32)), (self.dim_input,))
                        for ind in range(self.num_classes)]
                    )
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        if binary_classification:
            all_label_batches = tf.one_hot(all_label_batches, 2)
        else:
            all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

