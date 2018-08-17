import tensorflow as tf
import numpy as np

from meta_layers import conv2d, conv3d, batch_normalization


def test_conv2d():
    a = tf.reshape(tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]), (1, 3, 3, 1))
    b = tf.layers.conv2d(a, filters=2, kernel_size=(3, 3), strides=(1, 1), padding='SAME')

    W = tf.trainable_variables()[0]
    B = tf.trainable_variables()[1]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    b_np = sess.run(b)
    c_np = sess.run(conv2d(a, weights=W, bias=B, strides=(1, 1), padding='SAME'))

    assert(np.min(c_np == b_np))
    print('conv2d test passed successfully.')


def test_conv3d():
    a = tf.reshape(
        tf.constant([
            [1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
            [-1., -2., -3.], [-4., -5., -6.], [-7., -8., -9.],
            [11., 12., 13.], [14., 15., 16.], [17., 18., 19.],
        ]),
        (1, 3, 3, 3, 1)
    )
    b = tf.layers.conv3d(a, filters=2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='VALID')

    W = tf.trainable_variables()[0]
    B = tf.trainable_variables()[1]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    b_np = sess.run(b)
    c_np = sess.run(conv3d(a, weights=W, bias=B, strides=(1, 1, 1), padding='VALID', name='test'))

    assert(np.min(c_np == b_np))
    print('conv3d test passed successfully.')


def test_batch_norm():
    a = tf.placeholder(dtype=tf.float32, shape=(None, 5))

    b = tf.layers.batch_normalization(a, momentum=0.999, name='scope_name', training=False)

    # from tensorflow.contrib.layers.python.layers import batch_norm as bn
    # b = bn(a, scope='test')

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scope_name')
    variables = {variable.name: variable for variable in variables}

    c = batch_normalization(
        a,
        mean=variables['scope_name/moving_mean:0'],
        variance=variables['scope_name/moving_variance:0'],
        offset=variables['scope_name/beta:0'],
        name='scope_name'
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        b_np, c_np = sess.run((b, c), feed_dict={a: [[1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5]]})
        # c_np = sess.run(c, feed_dict={a: [[1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5]]})

        assert(np.min(c_np == b_np))
        b_np = sess.run(b, feed_dict={a: [[-1, -2, -3, -4, -5], [-0.1, -0.2, -0.3, -0.4, -0.5]]})
        c_np = sess.run(c, feed_dict={a: [[-1, -2, -3, -4, -5], [-0.1, -0.2, -0.3, -0.4, -0.5]]})
        assert (np.min(c_np == b_np))

    print('batch normalization test passed successfully.')

if __name__ == '__main__':
    # test_conv3d()
    test_batch_norm()