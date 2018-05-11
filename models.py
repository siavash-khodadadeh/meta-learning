import tensorflow as tf

from meta_layers import conv2d, dense


class NeuralNetwork(object):
    def __init__(self, input_layer, weights=None):
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
            self.dense = tf.layers.dense(self.flatten, 50, activation=tf.nn.relu, name='dense1')
            self.output = tf.layers.dense(self.dense, 5, activation=None, name='dense2')

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
            self.dense = dense(
                self.flatten,
                weights=weights['dense1/kernel:0'],
                bias=weights['dense1/bias:0'],
                activation=tf.nn.relu,
                name='dense1'
            )
            self.output = dense(
                self.dense,
                weights=weights['dense2/kernel:0'],
                bias=weights['dense2/bias:0'],
                activation=None,
                name='dense2'
            )


class ModelAgnosticMetaLearning(object):
    def __init__(
            self,
            model_cls,
            input_data,
            input_labels,
            input_validation,
            input_validation_labels,
            meta_learn_rate=0.0001,
            learning_rate=0.001,
    ):
        self.input_data = tf.reshape(input_data, (-1, 28, 28, 1))
        self.input_labels = tf.reshape(input_labels, (-1, 5))
        self.input_validation = tf.reshape(input_validation, (-1, 28, 28, 1))
        self.input_validation_labels = tf.reshape(input_validation_labels, (-1, 5))

        tf.summary.image('train', tf.reshape(self.input_data, (-1, 28, 28, 1)), max_outputs=25)
        tf.summary.image('validation', tf.reshape(self.input_validation, (-1, 28, 28, 1)), max_outputs=25)

        self.meta_learn_rate = meta_learn_rate
        self.learning_rate = learning_rate

        self.model_cls = model_cls
        with tf.variable_scope('model'):
            model = self.model_cls(self.input_data)
            self.model_out_train = model.output
            model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')

        with tf.variable_scope('loss'):
            self.train_loss = self.loss_function(self.input_labels, self.model_out_train)
            tf.summary.scalar('train_loss', self.train_loss)

        with tf.variable_scope('gradients'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.inner_train_op = optimizer.minimize(self.train_loss)
            self.grads = optimizer.compute_gradients(self.train_loss, var_list=model_variables)

            for grad_info in self.grads:
                tf.summary.histogram(grad_info[1].name, grad_info[0])

            self.updated_vars = {
                grad_info[1].name[6:]: grad_info[1] - self.learning_rate * grad_info[0]
                for grad_info in self.grads if grad_info[0] is not None
            }

        with tf.variable_scope('updated_model'):
            updated_model = self.model_cls(self.input_validation, self.updated_vars)
            self.model_out_validation = updated_model.output

        with tf.variable_scope('meta_loss', reuse=tf.AUTO_REUSE):
            self.meta_loss = self.loss_function(self.input_validation_labels, self.model_out_validation)
            tf.summary.scalar('meta_loss', self.meta_loss)
            # self.variance_loss = -tf.nn.moments(
            #     tf.cast(tf.argmax(self.model_out_validation, axis=1), tf.float32), axes=(0, )
            # )[1]
            # tf.summary.scalar('variance_loss', self.variance_loss)

        with tf.variable_scope('meta_optimizer'):
            meta_optimizer = tf.train.AdamOptimizer(learning_rate=self.meta_learn_rate)
            self.gradients = meta_optimizer.compute_gradients(self.meta_loss, var_list=model_variables)
            for grad_info in self.gradients:
                tf.summary.histogram(grad_info[1].name, grad_info[0])

            self.train_op = meta_optimizer.minimize(self.meta_loss, var_list=model_variables)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        self.file_writer = tf.summary.FileWriter('logs/2/', tf.get_default_graph())
        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(self.sess)

    def loss_function(self, y, y_hat):
        # return tf.reduce_sum(tf.square(y - y_hat))
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def save_model(self, step):
        self.saver.save(self.sess, 'saved_models/3/model', global_step=step)

    def load_model(self):
        self.saver.restore(self.sess, 'saved_models/3/model-5000')

    def meta_train(self):
        for it in range(100):
            self.sess.run(self.train_op)
