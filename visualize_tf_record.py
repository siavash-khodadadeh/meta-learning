import os

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


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
    resized_video = tf.reshape(decoded_video, shape=(-1, 112, 112, 3))

    # clip = resized_video[start_frame_number:start_frame_number + 16, :, :, :]
    clip = tf.reshape(resized_video, (-1, 112, 112, 3))

    return clip


def save_clip_as_gif(directory, saving_address):
    videos = []
    nrows = 4
    ncols = 4

    longest_clip_length = -1
    for record_address in sorted(os.listdir(directory))[:nrows * ncols]:
        sess = tf.Session()
        dataset = tf.data.TFRecordDataset([os.path.join(directory, record_address)])
        dataset = dataset.map(extract_video)
        iterator = dataset.make_one_shot_iterator()
        video = iterator.get_next()
        video_np = sess.run(video).reshape(1, -1, 112, 112, 3)
        longest_clip_length = max(longest_clip_length, video_np.shape[1])
        videos.append(video_np)

    fig = plt.figure(figsize=(16, 12), dpi=80)
    axes = []
    for row in range(nrows):
        for col in range(ncols):
            ax = fig.add_subplot(nrows, ncols, row * ncols + col + 1)
            axes.append(ax)

    def plt_ith_frame(i):
        for row in range(nrows):
            for col in range(ncols):
                video_clip = videos[row * ncols + col]
                axes[row * ncols + col].imshow(video_clip[0, min(i, video_clip.shape[1] - 1), :, :, :])

    #  This reference to the FuncAnimation should be here because apparently matplotlib cannot handle this
    #  if the garbage collector removes the reference to the FuncAnimation object!!!
    anim = FuncAnimation(fig, plt_ith_frame, frames=range(longest_clip_length), interval=50)
    anim.save(saving_address, writer='imagemagick')
    # plt.show()


if __name__ == '__main__':
    base_address = '/home/siavash/DIVA-TF-RECORDS/validation/'
    actions = os.listdir(base_address)
    for action in actions:
        action_address = os.path.join(base_address, action)
        saving_gif_address = '/home/siavash/diva-visualization/validation/{}.gif'.format(action)
        save_clip_as_gif(action_address, saving_gif_address)
