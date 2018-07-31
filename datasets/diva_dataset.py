import os

import numpy as np
import cv2
import h5py

from datasets.ucf101_data_generator import DataSetUtils


DIVA_DATASET_ADDRESS = '/home/siavash/DIVA-FewShot/'
DIVA_TFRECORD_DATASET_ADDRESS = '/home/siavash/DIVA-TF-RECORDS/'


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
    video = pre_process(video)
    return video


def create_tf_records_from_diva_h5_format(dataset_address, tf_record_address):
    for dataset_type in ('train', 'validation'):
        base_address = os.path.join(dataset_address, dataset_type)
        base_tf_address = os.path.join(tf_record_address, dataset_type)
        for action_name in os.listdir(base_address):
            for sample_name in os.listdir(os.path.join(base_address, action_name)):
                if sample_name == 'labels':
                    continue
                sample_address = os.path.join(base_address, action_name, sample_name)
                clip_address = sample_address
                clip = read_h5_file(clip_address)
                DataSetUtils.check_tf_directory(base_tf_address, action_name)
                DataSetUtils.write_tf_record(base_tf_address, clip, sample_name, action_name)


if __name__ == '__main__':
    create_tf_records_from_diva_h5_format(DIVA_DATASET_ADDRESS, DIVA_TFRECORD_DATASET_ADDRESS)



