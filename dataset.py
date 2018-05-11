import os

import numpy as np

from settings import DATASET_ADDRESS
from utils import load_image


class Alphabet(object):
    def __init__(self, name):
        self.name = name
        self.path = DATASET_ADDRESS + name + '/'
        self.num_chars = len(os.listdir(self.path))
        self.chars = []

        for file_address in sorted(os.listdir(self.path)):
            self.chars.append(Character(self, int(file_address[-2:])))

    def get_chars(self):
        return self.chars

    def __repr__(self):
        return self.name

    @staticmethod
    def all():
        alphabets = []
        for alphabet_name in os.listdir(DATASET_ADDRESS):
            alphabet = Alphabet(alphabet_name)
            alphabets.append(alphabet)

        return alphabets


class Character(object):
    def __init__(self, alphabet, char_number):
        self.alphabet = alphabet
        self.number = char_number
        fill = '0' if self.number < 10 else ''
        self.path = self.alphabet.path + 'character' + '{fill}{number}/'.format(fill=fill, number=self.number)
        self.sample_prefix = os.listdir(self.path)[0][:5]

    def load_sample(self, sample):
        fill = '0' if sample < 10 else ''
        image_path = self.sample_prefix + '{fill}{sample}.png'.format(fill=fill, sample=sample)
        image = load_image(self.path + image_path)
        return image

    def get_all_samples(self):
        all_samples = []
        for sample_num in range(1, 21):
            all_samples.append(self.load_sample(sample_num))
        return np.array(all_samples)

    @staticmethod
    def all():
        characters = []
        for alphabet in Alphabet.all():
            characters.extend(alphabet.get_chars())

        return characters

    def __repr__(self):
        return self.alphabet.name + '-character' + str(self.number)


class Dataset(object):
    def __init__(self, data, labels, num_concepts, num_shot_per_concept, shuffle=True):
        self.data = data
        self.labels = labels
        self.num_concepts = num_concepts
        self.num_shot_per_concept = num_shot_per_concept
        self.concepts_list = np.arange(self.num_concepts)
        if shuffle:
            self.shuffle_data()
        self._concept_index = 0
        self._within_concept_index = 0

    def shuffle_within_concepts(self):
        for row in range(self.num_concepts):
            begin = row * self.num_shot_per_concept
            end = begin + self.num_shot_per_concept
            np.random.shuffle(self.data[begin:end, :, :])

    def shuffle_concepts(self):
        np.random.shuffle(self.concepts_list)

    def shuffle_data(self):
        self.shuffle_within_concepts()
        self.shuffle_concepts()

    def next_batch(self, concept_size=10, concepts=None):
        # return concept_size * shot_per_concept of images and corresponding labels

        if concepts is None:
            concept_begin = self._concept_index
            concept_end = self._concept_index + concept_size
            if concept_end <= self.num_concepts:
                concepts = self.concepts_list[concept_begin:concept_end]
                self._concept_index = concept_end
            else:
                pre_concepts = self.concepts_list[concept_begin:]
                pre_concepts_len = len(pre_concepts)
                self.shuffle_data()
                self._concept_index = concept_size - pre_concepts_len
                next_concepts = self.concepts_list[:self._concept_index]
                concepts = np.concatenate((pre_concepts, next_concepts), axis=0)

        begin = self._within_concept_index
        end = begin + self.num_shot_per_concept

        data = []
        labels = []
        for concept in concepts:
            begin_index = concept * self.num_shot_per_concept + begin
            end_index = begin_index + end
            data.append(self.data[begin_index:end_index, :, :])
            labels.append(self.labels[begin_index:end_index, :])

        return np.concatenate(np.array(data), axis=0), np.concatenate(np.array(labels), axis=0)


def get_all_characters_images_and_labels(train_concepts=1000, num_shot_per_concept=5):
    train_images = []
    validation_images = []
    train_labels = []
    validation_labels = []
    test_images = []
    test_labels = []
    test_images_val = []
    test_labels_val = []
    label_counter = 0
    test_label_counter = 0
    added_train_concepts = 0
    added_test_concepts = 0

    for character in Character.all():
        all_samples = character.get_all_samples()

        if train_concepts > added_train_concepts:
            train_images.append(all_samples[:num_shot_per_concept, :, :])
            validation_images.append(all_samples[num_shot_per_concept:2 * num_shot_per_concept, :, :])
            for i in range(num_shot_per_concept):
                train_labels.append(label_counter)
                validation_labels.append(label_counter)

            added_train_concepts += 1
            label_counter += 1

        else:
            test_images.append(all_samples[:num_shot_per_concept, :, :])
            test_images_val.append(all_samples[num_shot_per_concept:2 * num_shot_per_concept, :, :])
            for i in range(num_shot_per_concept):
                test_labels.append(test_label_counter)
                test_labels_val.append(test_label_counter)

            added_test_concepts += 1
            test_label_counter += 1

    return {
        'train_images': np.concatenate(train_images, axis=0),
        'validation_images': np.concatenate(validation_images, axis=0),
        'test_images': np.concatenate(test_images, axis=0),
        'test_images_val': np.concatenate(test_images_val, axis=0),
        'train_labels': np.array(train_labels).reshape(-1, 1),
        'validation_labels': np.array(validation_labels).reshape(-1, 1),
        'test_labels': np.array(test_labels).reshape(-1, 1),
        'test_labels_val': np.array(test_labels_val).reshape(-1, 1),
        'num_train_concepts': added_train_concepts,
        'num_test_concepts': added_test_concepts,
    }


def load_datasets(num_shot_per_concept=5):
    data_info = get_all_characters_images_and_labels(num_shot_per_concept=num_shot_per_concept)
    train_images = data_info['train_images']
    train_labels = data_info['train_labels']
    num_train_concepts = data_info['num_train_concepts']
    validation_images = data_info['validation_images']
    validation_labels = data_info['validation_labels']
    test_images = data_info['test_images']
    test_labels = data_info['test_labels']
    test_images_val = data_info['test_images_val']
    test_labels_val = data_info['test_labels_val']
    num_test_concepts = data_info['num_test_concepts']

    train_dataset = Dataset(train_images, train_labels, num_train_concepts, num_shot_per_concept)
    validation_dataset = Dataset(validation_images, validation_labels, num_train_concepts, num_shot_per_concept)
    test_dataset = Dataset(test_images, test_labels, num_test_concepts, num_shot_per_concept)
    test_dataset_val = Dataset(test_images_val, test_labels_val, num_test_concepts, num_shot_per_concept)

    return train_dataset, validation_dataset, test_dataset, test_dataset_val
