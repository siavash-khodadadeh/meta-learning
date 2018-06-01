import tensorflow as tf

name_table = {
    'var_name/bc1': 'model/conv1/bias',
    'var_name/wc1': 'model/conv1/kernel',
    'var_name/bc2': 'model/conv2/bias',
    'var_name/wc2': 'model/conv2/kernel',
    'var_name/bc3a': 'model/conv3a/bias',
    'var_name/wc3a': 'model/conv3a/kernel',
    'var_name/bc3b': 'model/conv3b/bias',
    'var_name/wc3b': 'model/conv3b/kernel',
    'var_name/bc4a': 'model/conv4a/bias',
    'var_name/wc4a': 'model/conv4a/kernel',
    'var_name/bc4b': 'model/conv4b/bias',
    'var_name/wc4b': 'model/conv4b/kernel',
    'var_name/bc5a': 'model/conv5a/bias',
    'var_name/wc5a': 'model/conv5a/kernel',
    'var_name/bc5b': 'model/conv5b/bias',
    'var_name/wc5b': 'model/conv5b/kernel',
    'var_name/bd1': 'model/dense1/bias',
    'var_name/wd1': 'model/dense1/kernel',
    'var_name/bd2': 'model/dense2/bias',
    'var_name/wd2': 'model/dense2/kernel',
    'var_name/bout': 'model/out/bias',
    'var_name/wout': 'model/out/kernel',
}


BASE_FILE_ADDRESS = 'pretrained-model/conv3d_deepnetA_sport1m_iter_1900000_TF.model'

with tf.Graph().as_default(), tf.Session().as_default() as sess:
    new_vars = []
    vars = tf.contrib.framework.list_variables(BASE_FILE_ADDRESS)
    print(vars)
    for name, shape in vars:
        if name == 'global_step':
            continue

        new_name = name_table[name]
        if 'out' in new_name:
            continue
        v = tf.contrib.framework.load_variable(BASE_FILE_ADDRESS, name)

        new_vars.append(tf.Variable(v, name=new_name))

    print(new_vars)

    saver = tf.train.Saver(new_vars)
    sess.run(tf.global_variables_initializer())
    saver.save(sess, './MAML/sports1m_pretrained.model')
