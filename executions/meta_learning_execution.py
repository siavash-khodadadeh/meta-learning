import os
import random

import tensorflow as tf

from datasets.omniglot_dataset import get_omniglot_tf_record_dataset, \
    create_k_sample_per_action_iterative_omniglot_dataset
from datasets.tf_datasets import create_ucf101_data_feed_for_k_sample_per_action_iterative_dataset, \
    create_data_feed_for_train, create_diva_data_feed_for_k_sample_per_action_iterative_dataset_unique_class_each_batch
from models import ModelAgnosticMetaLearning, C3DNetwork, NeuralNetwork
import settings

from experiment_settings import RANDOM_SEED, DATASET, N, K, BATCH_SIZE, NUM_GPUS, NUM_ITERATIONS, META_LEARNING_RATE, \
    LEARNING_RATE, META_TRAIN, test_actions, diva_test_actions, STARTING_POINT_MODEL_ADDRESS, REPORT_AFTER_STEP, \
    SAVE_AFTER_STEP, META_TEST_STARTING_MODEL, NUM_META_TEST_ITERATIONS, SAVE_AFTER_META_TEST_STEP, \
    FIRST_OREDER_APPROXIMATION, BATCH_NORMALIZATION


def initialize():
    if test_actions is not None:
        execution_test_actions = test_actions[:BATCH_SIZE * NUM_GPUS]
    else:
        execution_test_actions = None

    if RANDOM_SEED != -1:
        random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

    model_dir = os.path.join(
        DATASET,
        'meta-train',
        'multiple-gpus',
        ('with' if FIRST_OREDER_APPROXIMATION else 'without') + '-first-order-approximation',
        ('with' if BATCH_NORMALIZATION else 'without') + '-batch-normalization',
        '{}-way-classifier'.format(N),
        '{}-shot'.format(K),
        'batch-size-{}'.format(BATCH_SIZE),
        'num-gpus-{}'.format(NUM_GPUS),
        'random-seed-{}'.format(RANDOM_SEED),
        'num-iterations-{}'.format(NUM_ITERATIONS),
        'meta-learning-rate-{}'.format(META_LEARNING_RATE),
        'learning-rate-{}'.format(LEARNING_RATE),
    )

    if META_TRAIN:
        log_dir = os.path.join(settings.BASE_LOG_ADDRESS, model_dir)
        saving_path = os.path.join(settings.SAVED_MODELS_ADDRESS, model_dir)
    else:
        log_dir = os.path.join(settings.BASE_LOG_ADDRESS, 'meta-test')
        saving_path = os.path.join(settings.SAVED_MODELS_ADDRESS, 'meta-test', 'model')

    if DATASET == 'ucf-101':
        base_address = settings.UCF101_TF_RECORDS_ADDRESS
    elif DATASET == 'diva':
        base_address = settings.DIVA_TRAIN_TF_RECORDS_ADDRESS
    elif DATASET == 'omniglot':
        base_address = settings.OMNIGLOT_TF_RECORD_ADDRESS
    else:
        base_address = settings.KINETICS_TF_RECORDS_ADDRESS

    if META_TRAIN:
        with tf.device('/cpu:0'):
            if DATASET == 'omniglot':
                with tf.variable_scope('data_reader'):
                    input_data_ph, input_labels_ph, val_data_ph, val_labels_ph, iterator, table = \
                        get_omniglot_tf_record_dataset(
                            num_classes=N,
                            num_samples_per_class=K,
                            meta_batch_size=1,
                        )

            else:
                input_data_ph, input_labels_ph, val_data_ph, val_labels_ph, iterator = create_data_feed_for_train(
                    base_address=base_address,
                    test_actions=execution_test_actions,
                    batch_size=BATCH_SIZE * NUM_GPUS,
                    k=K,
                    n=N,
                    random_labels=False
                )
    else:
        if DATASET == 'ucf-101' or DATASET == 'kinetics':
            print("test actiosn: ")
            print(execution_test_actions)

            input_data_ph, input_labels_ph, iterator, table = \
                create_ucf101_data_feed_for_k_sample_per_action_iterative_dataset(
                    dataset_address=base_address,
                    k=K,
                    batch_size=BATCH_SIZE * NUM_GPUS,
                    actions_include=execution_test_actions,
                )
            val_data_ph = input_data_ph
            val_labels_ph = input_labels_ph
        elif DATASET == 'diva':
            # input_data_ph, input_labels_ph, iterator = create_diva_data_feed_for_k_sample_per_action_iterative_dataset(
            #     dataset_address=base_address,
            #     k=K,
            #     batch_size=BATCH_SIZE * NUM_GPUS,
            # )
            input_data_ph, input_labels_ph, iterator, table = \
                create_diva_data_feed_for_k_sample_per_action_iterative_dataset_unique_class_each_batch(
                    dataset_address=base_address,
                    actions_include=None
                )

            val_data_ph = input_data_ph
            val_labels_ph = input_labels_ph
        else:
            input_data_ph, input_labels_ph, iterator, table = create_k_sample_per_action_iterative_omniglot_dataset(
                base_address,
                K,
                batch_size=BATCH_SIZE * NUM_GPUS
            )

            val_data_ph = input_data_ph
            val_labels_ph = input_labels_ph

    maml = ModelAgnosticMetaLearning(
        C3DNetwork,
        input_data_ph,
        input_labels_ph,
        val_data_ph,
        val_labels_ph,
        log_dir=log_dir,
        saving_path=saving_path,
        num_gpu_devices=NUM_GPUS,
        meta_learn_rate=META_LEARNING_RATE,
        learning_rate=LEARNING_RATE,
        log_device_placement=False,
        first_order_approximation=FIRST_OREDER_APPROXIMATION,
        num_classes=N,
        debug=False,
    )

    maml.sess.run(tf.tables_initializer())
    maml.sess.run(iterator.initializer)
    if not META_TRAIN:
        print(maml.sess.run(table.export()))

    return maml, os.path.join(settings.SAVED_MODELS_ADDRESS, model_dir)


if __name__ == '__main__':
    maml, loading_dir = initialize()
    if META_TRAIN:
        if STARTING_POINT_MODEL_ADDRESS:
            maml.load_model(path=STARTING_POINT_MODEL_ADDRESS, load_last_layer=False)
        maml.meta_train(
            num_iterations=NUM_ITERATIONS + 1,
            report_after_x_step=REPORT_AFTER_STEP,
            save_after_x_step=SAVE_AFTER_STEP
        )
    else:
        maml.load_model(os.path.join(loading_dir, META_TEST_STARTING_MODEL))
        maml.meta_test(NUM_META_TEST_ITERATIONS, save_model_per_x_iterations=SAVE_AFTER_META_TEST_STEP)
