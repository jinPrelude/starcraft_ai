from pysc2.lib import actions
import tensorflow as tf
import numpy as np

def postprocessing(s, x, y) :
    if np.ndim(s) > 1 :
        s = s[:,:,:,-1]
        s = np.reshape(s, (64, 64))

    action = actions.FUNCTIONS.no_op()

    if x != 0 and y != 0:
        if s[x][y] == 0:
            action = actions.FUNCTIONS.Move_screen("now", (x, y))

        elif s[x][y] == 1:
            action = actions.FUNCTIONS.select_point("select", (x, y))

        elif s[x][y] == 4:
            action = actions.FUNCTIONS.Attack_screen("now", (x, y))

    return action


class actorNetwork() :
    def __init__(self, sess, screen_size, action_dim, action_bound):
        self.sess = sess
        self.screen_size = screen_size
        self.action_dim = action_dim
        self.action_bound = action_bound


        with tf.variable_scope('actor') :
            self.inputs, self.out = self.create_actor_network()

    def create_actor_network(self):
        inputs = tf.placeholder(tf.float32, shape=[None, self.screen_size, self.screen_size, 4])

        conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.relu)

        conv_flat = tf.layers.flatten(conv3)

        w1 = tf.get_variable(name='w1', shape=[64 * 64, 500], dtype=tf.float32,
                             initializer=tf.random_uniform_initializer(-0.3, 0.3))
        l1 = tf.matmul(conv_flat, w1)
        l1 = tf.nn.relu(l1)

        w2 = tf.get_variable(name='w2', shape=[500, 2], dtype=tf.float32,
                             initializer=tf.random_uniform_initializer(-0.3, 0.3))
        l2 = tf.matmul(l1, w2)
        l2 = tf.nn.tanh(l2)
        l2 = tf.multiply(l2, self.action_bound)
        out = tf.to_int32(tf.add(l2, 32))

        return inputs, out

    def predict(self, s):
        s = s
        s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        a = self.sess.run(self.out, feed_dict={
            self.inputs : s
        })
        action = postprocessing(s, a[0][0], a[0][1])
        return action

class criticNetwork(object):

    def __init__(self):
        pass
