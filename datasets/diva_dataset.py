import os
import pickle

import numpy as np
import cv2
import h5py

from datasets.ucf101_data_generator import DataSetUtils
from settings import DIVA_DATASET_ADDRESS, DIVA_TFRECORD_DATASET_ADDRESS


REAL_CLASS_LABELS = {
    0: "Riding",
    1: "activity_standing",
    2: "vehicle_turning_right",
    3: "activity_walking",
    4: "vehicle_starting",
    5: "specialized_texting_phone",
    6: "vehicle_moving",
    7: "vehicle_stopping",
    8: "activity_carrying",
    9: "activity_gesturing",
    10: "Unloading",
    11: "Transport_HeavyCarry",
    12: "specialized_talking_phone",
    13: "Exiting",
    14: "Closing",
    15: "Misc",
    16: "vehicle_turning_left",
    17: "specialized_miscellaneous",
    18: "Interacts",
    19: "Entering",
    20: "Opening",
    21: "activity_running",
    22: "Open_Trunk",
    23: "Closing_Trunk",
    24: "vehicle_u_turn",
    25: "Person_Person_Interaction",
    26: "Loading",
    27: "Pull",
    28: "PickUp",
    29: "SetDown",
    30: "activity_sitting",
    31: "activity_crouching",
    32: "Talking",
    33: "PickUp_Person_Vehicle",
    34: "DropOff_Person_Vehicle",
    35: "Object_Transfer",
    36: "Drop",
    37: "specialized_using_tool",
    38: "Push",
    39: "specialized_throwing",
}


INTERESTING_CLASS_LABELS = {
    'specialized_texting_phone': 16,
    'specialized_talking_phone': 15,
    'Unloading': 12,
    'Transport_HeavyCarry': 11,
    'Talking': 10,
    'activity_carrying': 13,
    'Closing': 0,
    'vehicle_u_turn': 19,
    'Closing_Trunk': 1,
    'vehicle_turning_right': 18,
    'Entering': 2,
    'Exiting': 3,
    'Open_Trunk': 6,
    'activity_sitting': 14,
    'Interacts': 4,
    'vehicle_turning_left': 17,
    'Loading': 5,
    'Pull': 8,
    'Opening': 7,
    'Riding': 9,
}


def convert_to_5_fps(video):
    video_length = video.shape[0]
    sample_rate = min(int(video_length / 16), 5)
    return video[::sample_rate, :, :, :]


def resize_to(video, width=112, height=112):
    resized_video_frames = [cv2.resize(video[i, :, :, :], (width, height)) for i in range(video.shape[0])]
    resized_video_frames = np.concatenate(resized_video_frames).reshape(-1, width, height, 3)
    return resized_video_frames


def convert_from_bgr_to_rgb(video):
    video = video[..., ::-1]
    return video


def pre_process(video):
    video = convert_to_5_fps(video)
    video = convert_from_bgr_to_rgb(video)
    video = resize_to(video)
    return video


def read_h5_file(file_address):
    hf = h5py.File(file_address, 'r')
    video = hf.get('data')
    if video.shape[0] < 16:
        return None
    video = pre_process(video)
    return video


def extract_labels(labels_base_address, sample_name):
    label_number_part = sample_name[sample_name.index('_') + 1:sample_name.index('.')]
    label_number_text = label_number_part[label_number_part.index('_') + 1:]
    label_file = os.path.join(
        labels_base_address,
        label_number_text + '_actions_39.pkl'
    )
    with open(label_file, 'rb') as f:
        labels_of_sample = pickle.load(f, encoding='latin1')

    final_labels = [0] * 20
    indices = np.where(labels_of_sample == 1)[0]
    for index in indices:
        real_class_name = REAL_CLASS_LABELS[index]
        if real_class_name in INTERESTING_CLASS_LABELS.keys():
            final_labels[INTERESTING_CLASS_LABELS[real_class_name]] = 1

    final_labels = np.array(final_labels, dtype=np.uint8)
    return final_labels


def create_tf_records_from_diva_h5_format(dataset_address, tf_record_address):
    for dataset_type in ('train', 'validation'):
        base_address = os.path.join(dataset_address, dataset_type)
        base_tf_address = os.path.join(tf_record_address, dataset_type)
        for action_name in os.listdir(base_address):
            labels_base_address = os.path.join(base_address, action_name, 'labels')
            for sample_name in os.listdir(os.path.join(base_address, action_name)):
                if sample_name == 'labels':
                    continue
                sample_address = os.path.join(base_address, action_name, sample_name)
                clip_address = sample_address
                clip = read_h5_file(clip_address)
                if clip is not None:
                    DataSetUtils.check_tf_directory(base_tf_address, action_name)
                    labels_of_sample = extract_labels(labels_base_address, sample_name)

                    DataSetUtils.write_tf_record(
                        base_tf_address,
                        clip,
                        sample_name,
                        action_name,
                        labels=labels_of_sample
                    )


if __name__ == '__main__':
    create_tf_records_from_diva_h5_format(DIVA_DATASET_ADDRESS, DIVA_TFRECORD_DATASET_ADDRESS)



