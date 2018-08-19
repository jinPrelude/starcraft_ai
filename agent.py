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
    def __init__(self, sess, screen_size, action_dim, action_bound, tau):
        self.sess = sess
        self.screen_size = screen_size
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.tau = tau

        with tf.variable_scope('actor') :
            self.inputs, self.out = self.create_actor_network()

        self.actor_network_params = tf.trainable_variables()

        with tf.variable_scope('target_actor') :
            self.target_inputs, self.target_out = self.create_actor_network()

        self.target_actor_network_params = tf.trainable_variables()[len(self.actor_network_params):]

        self.trainable_params_num = len(self.actor_network_params) + len(self.target_actor_network_params)

        self.update_target_actor_network_params = \
            [self.target_actor_network_params[i].assign(tf.multiply(self.actor_network_params[i], self.tau) +
                                                  tf.multiply(self.target_actor_network_params[i], 1. - self.tau)) for i in
             range(len(self.target_actor_network_params))]



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

    def target_predict(self, s):
        s = s
        s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        a = self.sess.run(self.target_out, feed_dict={
            self.inputs : s
        })
        action = postprocessing(s, a[0][0], a[0][1])
        return action

    def update_target_actor_network(self):
        self.sess.run(self.update_target_actor_network_params)

    def trainable_params_num(self):
        return self.trainable_params_num

###############################################################

class criticNetwork(object):

    def __init__(self, sess, screen_size, actor_params_num, tau):
        self.sess =sess
        self.screen_size = screen_size
        self.actor_params_num = actor_params_num
        self.tau = tau

        with tf.variable_scope('critic') :
            self.inputs, self.out = self.create_critic_network()

        self.critic_network_params = tf.trainable_variables()[self.actor_params_num:]

        # Target critic network 생성
        with tf.variable_scope('target_critic_network'):
            self.target_inputs, self.target_out = self.create_critic_network()

        self.target_critic_network_params = tf.trainable_variables()[
                                            (len(self.critic_network_params) + self.actor_params_num):]

        self.update_target_critic_network_params = \
            [self.target_critic_network_params[i].assign(tf.multiply(self.critic_network_params[i], self.tau) \
                                                  + tf.multiply(self.target_critic_network_params[i], 1. - self.tau))
             for i in range(len(self.target_critic_network_params))]

    def create_critic_network(self):
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

        w2 = tf.get_variable(name='w2', shape=[500, 1], dtype=tf.float32,
                             initializer=tf.random_uniform_initializer(-0.3, 0.3))
        out = tf.matmul(l1, w2)

        return inputs, out

    def predict(self, s):

        s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        q = self.sess.run(self.out, feed_dict={
            self.inputs: s
        })
        return q

    def target_predict(self, s):

        s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        q = self.sess.run(self.target_out, feed_dict={
            self.inputs: s
        })
        return q

    def update_target_critic_network(self):
        self.sess.run(self.update_target_critic_network_params)