import tensorflow as tf
from keras.utils import to_categorical
import numpy as np

from data_generator import DataGenerator
from models import ModelAgnosticMetaLearning, NeuralNetwork


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


def train_maml():
    train = False
    num_classes = 5
    update_batch_size = 5
    meta_batch_size = 1

    data_generator = DataGenerator(update_batch_size * 2, meta_batch_size)
    with tf.variable_scope('data_reader'):
        image_tensor, label_tensor = data_generator.make_data_tensor(train=train)

    input_data_ph = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * update_batch_size, -1], name='train')
    input_labels_ph = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * update_batch_size, -1], name='labels')

    val_data_ph = tf.slice(image_tensor, [0, num_classes * update_batch_size, 0], [-1, -1, -1], name='validation')
    val_labels_ph = tf.slice(label_tensor, [0, num_classes * update_batch_size, 0], [-1, -1, -1], name='val_labels')

    maml = ModelAgnosticMetaLearning(NeuralNetwork, input_data_ph, input_labels_ph, val_data_ph, val_labels_ph)

    if train:
        print('start meta training.')

        for it in range(5000):
            maml.sess.run(maml.train_op)

            if it % 20 == 0:
                merged_summary, input_data, input_labels, valid_data, valid_labels = maml.sess.run(
                    (maml.merged, maml.input_data, maml.input_labels, maml.input_validation, maml.input_validation_labels)
                )
                # merged_summary, model_out_val, model_out_train, meta_grads = maml.sess.run(
                #     (maml.merged, maml.model_out_validation, maml.model_out_train, maml.gradients)
                # )
                # print(np.argmax(model_out_train, axis=1))
                # print(np.argmax(model_out_val, axis=1))
                maml.file_writer.add_summary(merged_summary, global_step=it)
                print(it)

        maml.save_model(step=it + 1)
    else:
        # maml.learn_new_experience()
        maml.load_model()
        print('just pre training')
        test_batch, test_batch_labels, test_val_batch, test_val_batch_labels = maml.sess.run(
            [maml.input_data, maml.input_labels, maml.input_validation, maml.input_validation_labels]
        )
        for any_batch in range(1):
            for it in range(10000):
                if it % 20 == 0:
                    print(it)
                    summary = maml.sess.run(maml.merged, feed_dict={
                        maml.input_data: test_batch,
                        maml.input_labels: test_batch_labels,
                        maml.input_validation: test_val_batch,
                        maml.input_validation_labels: test_val_batch_labels,
                    })
                    maml.file_writer.add_summary(summary, global_step=1000 + it)

                _, loss = maml.sess.run([maml.inner_train_op, maml.train_loss], feed_dict={
                    maml.input_data: test_batch,
                    maml.input_labels: test_batch_labels,
                    # maml.input_validation: test_val_batch,
                    # maml.input_validation_labels: test_val_batch_labels,
                })

            outputs, loss = maml.sess.run([maml.model_out_train, maml.train_loss], feed_dict={
                maml.input_data: test_val_batch,
                maml.input_labels: test_val_batch_labels,
            })

            # print(loss)
            outputs_np = np.argmax(outputs, axis=1)
            print(outputs_np)
            labels_np = np.argmax(test_val_batch_labels.reshape(-1, 5), axis=1)
            print(labels_np)

            print('accuracy:')
            acc_num = np.sum(outputs_np == labels_np)
            acc = acc_num / 25.
            print(acc_num)
            print(acc)

    print('done')


if __name__ == '__main__':
    import os.path
    if os.path.exists('./logs/2/'):
        import shutil
        shutil.rmtree('./logs/2/')

    train_maml()
