import matplotlib.pyplot as plt

from datasets.dataset import Alphabet, Character, get_all_characters_images_and_labels
from settings import DATASET_ADDRESS
from utils import load_image


def test_load_image():
    path = DATASET_ADDRESS + 'Angelic/' + 'character01/' + '0965_01.png'
    img = load_image(path)
    plt.imshow(img)
    plt.show()


def test_get_chars():
    alphabet = Alphabet('Angelic')
    for character in alphabet.get_chars():
        print(character)
        for sample in range(1, 21):
            img = character.load_sample(sample)
            plt.imshow(img)
            plt.show()


def test_character_get_all_samples():
    alphabet = Alphabet('Angelic')
    character = alphabet.get_chars()[5]
    samples = character.get_all_samples()
    plt.imshow(samples[6, :, :])
    plt.show()
    assert(samples.shape == (20, 105, 105))


def test_alphabet_get_all():
    for alphabet in Alphabet.all():
        print(alphabet)


def test_character_all():
    for character in Character.all():
        print(character)


def test_get_all_character_images():
    images, labels = get_all_characters_images_and_labels()
    print(images.shape)
    plt.imshow(images[10, :, :])
    plt.show()


if __name__ == '__main__':
    # test_load_image()
    # test_get_chars()
    # test_character_get_all_samples()
    # test_alphabet_get_all()
    # test_character_all()
    test_get_all_character_images()
