import os
import sys

import tensorflow as tf
import numpy as np

from models import ModelAgnosticMetaLearning, C3DNetwork
from settings import UCF101_TF_RECORDS_ADDRESS
import settings


LOG_DIR = os.path.join(settings.PROJECT_ADDRESS, 'logs/temp/')

SAVED_MODEL_ADDRESS = os.path.join(settings.SAVED_MODELS_ADDRESS, 'newton/-4')


TEST_ACTIONS = {
    'YoYo': 99,
    'WritingOnBoard': 98,
    'JumpRope': 45,
    'Nunchucks': 54,
    'HulaHoop': 41,
    'MoppingFloor': 53,
    'HorseRace': 39,
    'TaiChi': 89,
    'Hammering': 34,
    'BreastStroke': 17,
    'PlayingTabla': 64,
    'GolfSwing': 31,
    'CleanAndJerk': 19,
    'FrisbeeCatch': 29,
    'HandstandWalking': 36,
    'FloorGymnastics': 28,
    'BrushingTeeth': 18,
    'Haircut': 32,
    'Diving': 24,
    'Mixing': 52,
    'HighJump': 38,
    'BodyWeightSquats': 13,
    'PlayingSitar': 63,
    'Basketball': 6,
    'BoxingSpeedBag': 16,
    'HammerThrow': 33,
    'BaseballPitch': 5,
    'CliffDiving': 20,
    'Biking': 9,
    'BandMarching': 4,
    'TennisSwing': 90,
    'BabyCrawling': 2,
    'BlowingCandles': 12,
    'FieldHockeyPenalty': 27,
    'Rowing': 74,
    'TrampolineJumping': 92,
    'PlayingDaf': 58,
    'Archery': 1,
    'BasketballDunk': 7,
    'BenchPress': 8,
    'ApplyLipstick': 0,
    'Skiing': 79,
    'BalanceBeam': 3,
    'CricketShot': 22,
    'Billiards': 10,
    'BlowDryHair': 11,
    'HandstandPushups': 35,
    'Drumming': 25,
    'PommelHorse': 67,
    'IceDancing': 42,
    'ThrowDiscus': 91,
    'Bowling': 14,
    'RockClimbingIndoor': 72,
    'WallPushups': 97,
    'BoxingPunchingBag': 15,
    'WalkingWithDog': 96,
    'JumpingJack': 46,
    'CricketBowling': 21,
    'Knitting': 48,
    'Kayaking': 47,
    'PlayingCello': 57,
    'PlayingPiano': 62,
    'PlayingViolin': 65,
    'HeadMassage': 37,
    'ParallelBars': 55,
    'PizzaTossing': 56,
    'MilitaryParade': 51,
    'PlayingDhol': 59,
    'PlayingFlute': 60,
    'Lunges': 50,
    'PlayingGuitar': 61,
    'PoleVault': 66,
    'PullUps': 68,
    'JavelinThrow': 43,
    'SumoWrestling': 85,
    'SkyDiving': 81,
    'Punch': 69,
    'RopeClimbing': 73,
    'Rafting': 71,
    'SoccerJuggling': 82,
    'CuttingInKitchen': 23,
    'LongJump': 49,
    'SalsaSpin': 75,
    'Swing': 87,
    'FrontCrawl': 30,
    'ShavingBeard': 76,
    'Shotput': 77,
    'PushUps': 70,
    'SkateBoarding': 78,
    'Skijet': 80,
    'Fencing': 26,
    'VolleyballSpiking': 95,
    'SoccerPenalty': 83,
    'StillRings': 84,
    'JugglingBalls': 44,
    'HorseRiding': 40,
    'Surfing': 86,
    'TableTennisShot': 88,
    'Typing': 93,
    'UnevenBars': 94,
}


TEST_ACTIONS = {
    'YoYo': 79,
    'UnevenBars': 74,
    'Typing': 73,
    'TrampolineJumping': 72,
    'TableTennisShot': 68,
    'Surfing': 66,
    'StillRings': 64,
    'SoccerPenalty': 63,
    'VolleyballSpiking': 75,
    'Skijet': 60,
    'Skiing': 59,
    'SkateBoarding': 58,
    'Shotput': 57,
    'ShavingBeard': 56,
    'SalsaSpin': 55,
    'RockClimbingIndoor': 52,
    'SoccerJuggling': 62,
    'Rafting': 51,
    'RopeClimbing': 53,
    'Punch': 49,
    'SumoWrestling': 65,
    'PullUps': 48,
    'PoleVault': 46,
    'TaiChi': 69,
    'HorseRace': 19,
    'HighJump': 18,
    'CuttingInKitchen': 3,
    'Swing': 67,
    'HandstandWalking': 16,
    'FloorGymnastics': 8,
    'WritingOnBoard': 78,
    'PommelHorse': 47,
    'HandstandPushups': 15,
    'ThrowDiscus': 71,
    'FrontCrawl': 10,
    'Rowing': 54,
    'Hammering': 14,
    'GolfSwing': 11,
    'IceDancing': 22,
    'PushUps': 50,
    'CliffDiving': 0,
    'FrisbeeCatch': 9,
    'HeadMassage': 17,
    'CricketShot': 2,
    'HammerThrow': 13,
    'SkyDiving': 61,
    'PizzaTossing': 36,
    'WalkingWithDog': 76,
    'FieldHockeyPenalty': 7,
    'PlayingDaf': 38,
    'Drumming': 5,
    'Fencing': 6,
    'JavelinThrow': 23,
    'Mixing': 32,
    'Diving': 4,
    'HulaHoop': 21,
    'MoppingFloor': 33,
    'JugglingBalls': 24,
    'JumpRope': 25,
    'Nunchucks': 34,
    'CricketBowling': 1,
    'JumpingJack': 26,
    'TennisSwing': 70,
    'Kayaking': 27,
    'WallPushups': 77,
    'Knitting': 28,
    'Lunges': 30,
    'ParallelBars': 35,
    'PlayingDhol': 39,
    'LongJump': 29,
    'PlayingPiano': 42,
    'PlayingCello': 37,
    'PlayingFlute': 40,
    'PlayingGuitar': 41,
    'HorseRiding': 20,
    'PlayingSitar': 43,
    'Haircut': 12,
    'PlayingTabla': 44,
    'MilitaryParade': 31,
    'PlayingViolin': 45,
}


# TEST_ACTIONS = {
#     'StillRings': 4,
#     'SoccerPenalty': 3,
#     'SoccerJuggling': 2,
#     'SkyDiving': 1,
#     'Skijet': 0,
# }


def extract_video(example):
    features = {
        'task': tf.FixedLenFeature([], tf.string),
        'len': tf.FixedLenFeature([], tf.int64),
        'video': tf.FixedLenFeature([], tf.string),
    }

    parsed_example = tf.parse_single_example(example, features)
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


def evaluate():
    with tf.variable_scope('train_data'):
        input_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        input_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, len(TEST_ACTIONS)])
        tf.summary.image('train', input_data_ph[:, 0, :, :, :], max_outputs=25)

    with tf.variable_scope('validation_data'):
        val_data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 16, 112, 112, 3])
        val_labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, len(TEST_ACTIONS)])
        tf.summary.image('validation', val_data_ph[:, 0, :, :, :], max_outputs=25)

    maml = ModelAgnosticMetaLearning(
        C3DNetwork,
        input_data_ph,
        input_labels_ph,
        val_data_ph,
        val_labels_ph,
        num_gpu_devices=1,
        log_dir=LOG_DIR,
        learning_rate=0.001,
        log_device_placement=False,
        saving_path=None,
        num_classes=len(TEST_ACTIONS),
    )

    maml.load_model(path=SAVED_MODEL_ADDRESS)

    correct = 0
    count = 0

    class_labels_couners = []

    for action in sorted(TEST_ACTIONS.keys()):
        class_label_counter = [0] * len(TEST_ACTIONS)
        print(action)
        for file_address in os.listdir(os.path.join(UCF101_TF_RECORDS_ADDRESS, action)):
            tf_record_address = os.path.join(UCF101_TF_RECORDS_ADDRESS, action, file_address)
            dataset = tf.data.TFRecordDataset([tf_record_address])
            dataset = dataset.map(extract_video)
            iterator = dataset.make_one_shot_iterator()
            video = iterator.get_next()
            video_np = maml.sess.run(video)
            video_np = video_np.reshape(1, 16, 112, 112, 3)

            outputs = maml.sess.run(maml.model.output, feed_dict={
                maml.input_data: video_np,
            })

            label = np.argmax(outputs)

            if label == TEST_ACTIONS[action]:
                correct += 1

            count += 1
            class_label_counter[label] += 1

        print(class_label_counter)
        print(np.argmax(class_label_counter))
        class_labels_couners.append(class_label_counter)
        sys.stdout.flush()

    print('Accuracy: ')
    print(float(correct) / count)
    print(count)
    print(correct)

    confusion_matrix = np.array(class_labels_couners, dtype=np.float32).transpose()
    print('\n\n')
    print('confusion matrix')
    print(confusion_matrix)
    print('\n\n')
    columns_sum = np.sum(confusion_matrix, axis=0)
    rows_sum = np.sum(confusion_matrix, axis=1)

    counter = 0
    for action in sorted(TEST_ACTIONS.keys()):
        print(action)
        recall = confusion_matrix[counter][counter] / rows_sum[counter]
        precision = confusion_matrix[counter][counter] / columns_sum[counter]
        f1_score = 2 * precision * recall / (precision + recall)
        print('F1 Score: ')
        print(f1_score)
        counter += 1


if __name__ == '__main__':
    evaluate()
