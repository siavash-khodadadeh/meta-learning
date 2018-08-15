import os

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from meta_layers import conv2d, dense, conv3d
from utils import average_gradients


class NeuralNetwork(object):
    def __init__(self, input_layer, weights=None, num_classes=5):
        if weights is None:
            self.conv1 = tf.layers.conv2d(
                input_layer,
                64,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=tf.nn.relu,
                name='conv1',
            )
            self.maxpool1 = tf.layers.max_pooling2d(self.conv1, pool_size=(2, 2), strides=(1, 1))

            self.conv2 = tf.layers.conv2d(
                self.maxpool1,
                64,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=tf.nn.relu,
                name='conv2',
            )
            self.maxpool2 = tf.layers.max_pooling2d(self.conv2, pool_size=(2, 2), strides=(1, 1))

            self.conv3 = tf.layers.conv2d(
                self.maxpool2,
                64,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=tf.nn.relu,
                name='conv3',
            )
            self.maxpool3 = tf.layers.max_pooling2d(self.conv3, pool_size=(2, 2), strides=(1, 1))

            self.conv4 = tf.layers.conv2d(
                self.maxpool3,
                64,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=tf.nn.relu,
                name='conv4',
            )
            self.maxpool4 = tf.layers.max_pooling2d(self.conv4, pool_size=(2, 2), strides=(1, 1))

            self.flatten = tf.layers.flatten(self.maxpool4)
            self.output = tf.layers.dense(self.flatten, num_classes, activation=None, name='dense1')

        else:
            self.conv1 = conv2d(
                input_layer,
                weights=weights['conv1/kernel:0'],
                bias=weights['conv1/bias:0'],
                strides=(1, 1),
                activation=tf.nn.relu,
                name='conv1'
            )
            self.maxpool1 = tf.layers.max_pooling2d(self.conv1, pool_size=(2, 2), strides=(1, 1))

            self.conv2 = conv2d(
                self.maxpool1,
                weights=weights['conv2/kernel:0'],
                bias=weights['conv2/bias:0'],
                strides=(1, 1),
                activation=tf.nn.relu,
                name='conv2'
            )
            self.maxpool2 = tf.layers.max_pooling2d(self.conv2, pool_size=(2, 2), strides=(1, 1))

            self.conv3 = conv2d(
                self.maxpool2,
                weights=weights['conv3/kernel:0'],
                bias=weights['conv3/bias:0'],
                strides=(1, 1),
                activation=tf.nn.relu,
                name='conv3'
            )
            self.maxpool3 = tf.layers.max_pooling2d(self.conv3, pool_size=(2, 2), strides=(1, 1))

            self.conv4 = conv2d(
                self.maxpool3,
                weights=weights['conv4/kernel:0'],
                bias=weights['conv4/bias:0'],
                strides=(1, 1),
                activation=tf.nn.relu,
                name='conv4'
            )
            self.maxpool4 = tf.layers.max_pooling2d(self.conv4, pool_size=(2, 2), strides=(1, 1))

            self.flatten = tf.layers.flatten(self.maxpool4)
            self.output = dense(
                self.flatten,
                weights=weights['dense1/kernel:0'],
                bias=weights['dense1/bias:0'],
                activation=None,
                name='dense1'
            )

    def get_last_meta_layer(self):
        return self.flatten


class C3DNetwork(object):
    def __init__(self, input_layer, weights=None, num_classes=None):
        if weights is None:
            self.conv1 = tf.layers.conv3d(
                input_layer,
                64,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv1',
            )
            self.maxpool1 = tf.layers.max_pooling3d(self.conv1, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME')

            self.conv2 = tf.layers.conv3d(
                self.maxpool1,
                128,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv2',
            )
            self.maxpool2 = tf.layers.max_pooling3d(self.conv2, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME')

            self.conv3a = tf.layers.conv3d(
                self.maxpool2,
                256,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv3a',
            )
            self.conv3b = tf.layers.conv3d(
                self.conv3a,
                256,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv3b',
            )
            self.maxpool3 = tf.layers.max_pooling3d(self.conv3b, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME')

            self.conv4a = tf.layers.conv3d(
                self.maxpool3,
                512,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv4a',
            )
            self.conv4b = tf.layers.conv3d(
                self.conv4a,
                512,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv4b',
            )
            self.maxpool4 = tf.layers.max_pooling3d(self.conv4b, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME')

            self.conv5a = tf.layers.conv3d(
                self.maxpool4,
                512,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv5a',
            )
            self.conv5b = tf.layers.conv3d(
                self.conv5a,
                512,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv5b',
            )
            self.maxpool5 = tf.layers.max_pooling3d(self.conv5b, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME')

            self.transpose = tf.transpose(self.maxpool5, (0, 1, 4, 2, 3))
            self.flatten = tf.layers.flatten(self.transpose)
            self.dense = tf.layers.dense(self.flatten, 4096, activation=tf.nn.relu, name='dense1')
            self.dense2 = tf.layers.dense(self.dense, 4096, activation=tf.nn.relu, name='dense2')
            self.output = tf.layers.dense(self.dense2, num_classes, activation=None, name='dense3')
        else:
            self.conv1 = conv3d(
                input_layer,
                weights=weights['conv1/kernel:0'],
                bias=weights['conv1/bias:0'],
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv1'
            )
            self.maxpool1 = tf.layers.max_pooling3d(self.conv1, pool_size=(1, 2, 2), strides=(1, 2, 2))
            self.conv2 = conv3d(
                self.maxpool1,
                weights=weights['conv2/kernel:0'],
                bias=weights['conv2/bias:0'],
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv2'
            )
            self.maxpool2 = tf.layers.max_pooling3d(self.conv2, pool_size=(2, 2, 2), strides=(2, 2, 2))
            self.conv3a = conv3d(
                self.maxpool2,
                weights=weights['conv3a/kernel:0'],
                bias=weights['conv3a/bias:0'],
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv3a'
            )
            self.conv3b = conv3d(
                self.conv3a,
                weights=weights['conv3b/kernel:0'],
                bias=weights['conv3b/bias:0'],
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv3b'
            )
            self.maxpool3 = tf.layers.max_pooling3d(self.conv3b, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME')

            self.conv4a = conv3d(
                self.maxpool3,
                weights=weights['conv4a/kernel:0'],
                bias=weights['conv4a/bias:0'],
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv4a'
            )
            self.conv4b = conv3d(
                self.conv4a,
                weights=weights['conv4b/kernel:0'],
                bias=weights['conv4b/bias:0'],
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv4b'
            )
            self.maxpool4 = tf.layers.max_pooling3d(self.conv4b, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME')
            self.conv5a = conv3d(
                self.maxpool4,
                weights=weights['conv5a/kernel:0'],
                bias=weights['conv5a/bias:0'],
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv5a'
            )
            self.conv5b = conv3d(
                self.conv5a,
                weights=weights['conv5b/kernel:0'],
                bias=weights['conv5b/bias:0'],
                strides=(1, 1, 1),
                activation=tf.nn.relu,
                padding='SAME',
                name='conv5b'
            )
            self.maxpool5 = tf.layers.max_pooling3d(self.conv5b, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME')

            self.transpose = tf.transpose(self.maxpool5, (0, 1, 4, 2, 3))
            self.flatten = tf.layers.flatten(self.transpose)
            self.dense = dense(
                self.flatten,
                weights=weights['dense1/kernel:0'],
                bias=weights['dense1/bias:0'],
                activation=tf.nn.relu,
                name='dense1'
            )
            self.dense2 = dense(
                self.dense,
                weights=weights['dense2/kernel:0'],
                bias=weights['dense2/bias:0'],
                activation=tf.nn.relu,
                name='dense2'
            )
            self.output = dense(
                self.dense2,
                weights=weights['dense3/kernel:0'],
                bias=weights['dense3/bias:0'],
                activation=None,
                name='dense3'
            )


class ModelAgnosticMetaLearning(object):
    def __init__(
            self,
            model_cls,
            input_data_ph,
            input_labels_ph,
            input_validation_ph,
            input_validation_labels_ph,
            log_dir,
            saving_path,
            meta_learn_rate=0.00001,
            learning_rate=0.0001,
            num_gpu_devices=None,
            debug=False,
            log_device_placement=True,
            num_classes=None,
    ):
        self.devices = self._get_gpu_devices(num_gpu_devices)
        self.num_gpu_devices = len(self.devices)
        self.model_cls = model_cls
        self.meta_learn_rate = self.get_exponential_decay_learning_rate(meta_learn_rate)
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.meta_optimizer = tf.train.AdamOptimizer(learning_rate=self.meta_learn_rate)

        # self.input_data_ph = input_data_ph
        # self.input_labels_ph = input_labels_ph
        # self.input_validation_ph = input_validation_ph
        # self.input_validation_labels_ph = input_validation_labels_ph
        #
        # self.input_data = tf.placeholder(dtype=tf.float32, shape=(5, 28, 28, 1))
        # self.input_labels = tf.placeholder(dtype=tf.float32, shape=(5, 5))
        # self.input_validation = tf.placeholder(dtype=tf.float32, shape=(5, 28, 28, 1))
        # self.input_validation_labels = tf.placeholder(dtype=tf.float32, shape=(5, 5))

        self.input_data = input_data_ph
        self.input_labels = input_labels_ph
        self.input_validation = input_validation_ph
        self.input_validation_labels = input_validation_labels_ph

        self.grads, self.inner_loss = self._create_inner_model_part(self.input_data, self.input_labels)
        self.inner_train_op = self._define_inner_train_op(self.inner_loss)

        # grads_stopped = [(tf.stop_gradient(grad_and_var[0]), grad_and_var[1]) for grad_and_var in self.grads]

        updated_vars = self._compute_updated_vars(self.grads)
        self.meta_loss = self._create_meta_part(
            self.input_validation,
            self.input_validation_labels,
            updated_vars
        )

        with tf.variable_scope('meta_optimizer'):
            self.train_op = self.meta_optimizer.minimize(self.meta_loss, var_list=self.model_variables)

        # with tf.variable_scope('train_data'):
        #     tf.summary.image('train', self.input_data, max_outputs=25)
        #
        # with tf.variable_scope('validation_data'):
        #     tf.summary.image('validation', self.input_validation, max_outputs=25)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        self.log_dir = self._create_log_dir(log_dir)
        self.file_writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
        self.merged = tf.summary.merge_all()

        self.saving_path = saving_path
        self.saver = tf.train.Saver()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        if debug:
            self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "SSH:6000")

        self.sess.run(tf.global_variables_initializer())

    def _compute_updated_vars(self, averaged_inner_gradients):
        updated_vars = {}
        for grad_info in averaged_inner_gradients:
            if grad_info[0] is not None:
                updated_vars[grad_info[1].name[6:]] = grad_info[1] - self.learning_rate * grad_info[0]
            else:
                updated_vars[grad_info[1].name[6:]] = grad_info[1]

            # self.inner_train_ops.append(tf.assign(grad_info[1], updated_vars[grad_info[1].name[6:]]))
        return updated_vars

    def _define_inner_train_op(self, loss):
        return self.optimizer.minimize(loss)

    def _get_gpu_devices(self, num_gpu_devices):
        return ['/gpu:{}'.format(gpu_id) for gpu_id in range(num_gpu_devices)]

    def get_exponential_decay_learning_rate(self, initial_learning_rate):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 1000, 0.96, staircase=True)
        return learning_rate

    def loss_function(self, labels, logits):
        return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        # return tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits, weights=19 * labels + 1)

        # return tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

    def neural_loss_function(self, labels, logits):
        with tf.variable_scope('neural_loss', reuse=tf.AUTO_REUSE):
            loss_input = tf.concat((labels, logits), axis=1)
            print('loss input shape: ')
            print(loss_input.shape)
            loss_dense_1 = tf.layers.dense(loss_input, units=10, activation=tf.nn.relu, name='loss_dense_1')
            loss_dense_2 = tf.layers.dense(loss_dense_1, units=10, activation=tf.nn.relu, name='loss_dense_2')
            loss = tf.layers.dense(loss_dense_2, units=1, activation=None, name='loss_out')

            return tf.norm(loss)

    def save_model(self, path, step):
        if path[len(path) - 1] != '/':
            path += '/'
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError as exc:  # Guard against race condition
                raise exc

        self.saver = tf.train.Saver()
        self.saver.save(self.sess, path, global_step=step)

    def load_model(self, path, load_last_layer=True):
        if not load_last_layer:
            model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
            self.saver = tf.train.Saver(var_list=model_variables[:-2])

        self.saver.restore(self.sess, path)

    def meta_train(self, num_iterations, report_after_x_step, save_after_x_step):
        for it in range(num_iterations):
            img, lbl, val_img, val_lbl = self.sess.run(
                (self.input_data, self.input_labels, self.input_validation, self.input_validation_labels)
            )

            # if it % 100 == 0:
            #     import matplotlib.pyplot as plt
            #     fig, axes = plt.subplots(2, 5)
            #     for i in range(5):
            #         axes[0, i].imshow(img[i, ...].reshape(28, 28))
            #         axes[0, i].set_title(lbl[i, ...])
            #         axes[1, i].imshow(val_img[i, ...].reshape(28, 28))
            #         axes[1, i].set_title(val_lbl[i, ...])
            #     plt.show()

            feed_dict = {
                self.input_data: img,
                self.input_labels: lbl,
                self.input_validation: val_img,
                self.input_validation_labels: val_lbl,
            }

            inner_loss_value, meta_loss_value, merged_summary, _ = self.sess.run(
                (self.inner_loss, self.meta_loss, self.merged, self.train_op),
                feed_dict=feed_dict
            )

            if it % report_after_x_step == 0:
                self.file_writer.add_summary(merged_summary, global_step=it)
                print(it)
                print('inner loss value:')
                print(inner_loss_value)
                print('meta loss value:')
                print(meta_loss_value)


            if it % save_after_x_step == 0 and it != 0:
                self.save_model(path=self.saving_path, step=it)

    def meta_test(self, num_iterations, feed_data=None, save_model_per_x_iterations=20):
        for it in range(num_iterations):
            print(it)
            if feed_data is None:
                _, merged_summary = self.sess.run((self.inner_train_op, self.merged))
            else:
                _, merged_summary = self.sess.run((self.inner_train_op, self.merged), feed_dict=feed_data)

            self.file_writer.add_summary(merged_summary, global_step=it)
            if it % save_model_per_x_iterations == 0 and it != 0:
                self.save_model(path=self.saving_path, step=it)

    def evaluate(self, input_data=None):
        if input_data is None:
            outputs, real_labels = self.sess.run((self.model.output, self.input_labels))
        else:
            outputs, real_labels = self.sess.run((self.model.output, self.input_labels), feed_dict={
                self.input_data: input_data
            })
        import numpy as np
        print(outputs)
        print(real_labels)
        print(np.argmax(outputs, 1))
        print(np.argmax(real_labels, 1))
        return outputs

    def _split_data_between_devices(self):
        num_gpu_devices = len(self.devices)

        input_data_splits = tf.split(self.input_data, num_gpu_devices)
        input_labels_splits = tf.split(self.input_labels, num_gpu_devices)
        input_validation_splits = tf.split(self.input_validation, num_gpu_devices)
        input_validation_labels_splits = tf.split(self.input_validation_labels, num_gpu_devices)
        return input_data_splits, input_labels_splits, input_validation_splits, input_validation_labels_splits

    def _create_model(self, input_data):
        return self.model_cls(input_data, num_classes=self.num_classes)

    def _create_inner_model_part(self, input_data, input_labels):
        with tf.variable_scope('model'):
            self.model = self._create_model(input_data)
            model_out_train = self.model.output
            self.model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')

        with tf.variable_scope('loss'):
            train_loss = self.loss_function(input_labels, model_out_train)
            tf.summary.scalar('train_loss', train_loss)

        with tf.variable_scope('gradients'):
            grads = self._compute_inner_gradients(train_loss)

        return grads, train_loss

    def _compute_inner_gradients(self, train_loss):
        grads = self.optimizer.compute_gradients(
            train_loss,
            var_list=self.model_variables,
        )
        return grads

    def _create_meta_part(self, input_validation, input_validation_labels, updated_vars):
        with tf.variable_scope('updated_model'):
            updated_model = self.model_cls(input_validation, weights=updated_vars)
            model_out_validation = updated_model.output

        with tf.variable_scope('meta_loss'):
            meta_loss = self.loss_function(input_validation_labels, model_out_validation)
            tf.summary.scalar('meta_loss', meta_loss)

        return meta_loss

    def _create_log_dir(self, log_dir):
        if os.path.exists(log_dir):
            experiment_num = str(len(os.listdir(log_dir)))
        else:
            experiment_num = '0'
        return os.path.join(log_dir, experiment_num) + '/'


class ProgressiveModelAgnosticMetaLearning(ModelAgnosticMetaLearning):
    def __init__(
        self,
        model_cls,
        input_data_ph,
        input_labels_ph,
        input_validation_ph,
        input_validation_labels_ph,
        log_dir,
        saving_path,
        meta_learn_rate=0.00001,
        learning_rate=0.0001,
        num_gpu_devices=None,
        debug=False,
        log_device_placement=True,
    ):
        super(ProgressiveModelAgnosticMetaLearning, self).__init__(
            model_cls,
            input_data_ph,
            input_labels_ph,
            input_validation_ph,
            input_validation_labels_ph,
            log_dir,
            saving_path,
            meta_learn_rate,
            learning_rate,
            num_gpu_devices,
            debug,
            log_device_placement,
            num_classes=1,
        )
        self.model_outputs = []

    def _compute_inner_gradients(self, train_loss):
        grads = self.optimizer.compute_gradients(
            train_loss,
            var_list=self.model_variables[-4:],
            colocate_gradients_with_ops=True
        )
        self.inner_grads.append(grads)

    def _compute_updated_vars(self, averaged_inner_gradients):
        updated_vars = {var.name[6:]: var for var in self.model_variables}
        for grad_info in averaged_inner_gradients:
            if grad_info[0] is not None:
                updated_vars[grad_info[1].name[6:]] = grad_info[1] - self.learning_rate * grad_info[0]
            else:
                updated_vars[grad_info[1].name[6:]] = grad_info[1]

            self.inner_train_ops.append(tf.assign(grad_info[1], updated_vars[grad_info[1].name[6:]]))
        return updated_vars

    def learn_new_concept(self, instances, labels, iterations=1):
        layer_number = len(self.model_outputs)
        last_meta_layer = self.model.get_last_meta_layer()

        with tf.variable_scope('learned_layer_{}'.format(layer_number)):
            dense1 = tf.layers.dense(last_meta_layer, units=50, activation=tf.nn.relu, name='dense_1')
            dense2 = tf.layers.dense(dense1, 1, activation=tf.nn.relu, name='dense_2')

        self.model_outputs.append(dense2)

        layer_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='learned_layer_{}'.format(layer_number)
        )
        loss = self.loss_function(self.input_labels, self.model_outputs[-1])

        assign_op1 = tf.assign(layer_variables[0], self.model_variables[-4])
        assign_op2 = tf.assign(layer_variables[1], self.model_variables[-3])
        assign_op3 = tf.assign(layer_variables[2], self.model_variables[-2])
        assign_op4 = tf.assign(layer_variables[3], self.model_variables[-1])
        op = self.optimizer.minimize(loss, var_list=layer_variables, colocate_gradients_with_ops=True)

        self._initialize_uninitialized()
        self.sess.run((assign_op1, assign_op2, assign_op3, assign_op4))
        for it in range(iterations):
            _, merged_summary = self.sess.run((op, self.merged), feed_dict={
                self.input_data: instances,
                self.input_labels: labels
            })
            self.file_writer.add_summary(merged_summary, global_step=it + 1)

    def evaluate_progressive(self, input_data=None):
        assert input_data is not None
        outputs, merged_summary = self.sess.run((self.model_outputs, self.merged), feed_dict={
            self.input_data: input_data
        })
        self.file_writer.add_summary(merged_summary, global_step=0)
        return outputs

    def _initialize_uninitialized(self):
        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        print([str(i.name) for i in not_initialized_vars])  # only for testing
        if len(not_initialized_vars):
            self.sess.run(tf.variables_initializer(not_initialized_vars))

    def loss_function(self, labels, logits):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
