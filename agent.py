from pysc2.lib import actions
import tensorflow as tf
import numpy as np

def postprocessing(s, x, y) :

    s = s[:,:,:,-1]
    dim = np.shape(s)[0]
    # 1개 들어올때만 생각했고 batch로 들어올때 생각을 못함

    y_i = []
    for i in range(dim) :
        action = actions.FUNCTIONS.no_op()
        if x != 0 and y != 0:
            if s[i][x][y] == 0:
                action = actions.FUNCTIONS.Move_screen("now", (x, y))

            elif s[i][x][y] == 1:
                action = actions.FUNCTIONS.select_point("select", (x, y))

            elif s[i][x][y] == 4:
                action = actions.FUNCTIONS.Attack_screen("now", (x, y))
        y_i.append(action)
    print('end')
    if dim == 1 :
        return y_i[0]
    else :
        return y_i


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
        #s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        out = self.sess.run(self.out, feed_dict={
            self.inputs : s
        })
        action = postprocessing(s, out[0][0], out[0][1])
        return action, out

    def target_predict(self, s):
        #s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        out = self.sess.run(self.target_out, feed_dict={
            self.inputs : s
        })
        action = postprocessing(s, out[0][0], out[0][1])
        return action, out[0]

    def update_target_actor_network(self):
        self.sess.run(self.update_target_actor_network_params)

    def get_trainable_params_num(self):
        return self.trainable_params_num

###############################################################

class criticNetwork(object):

    def __init__(self, sess, screen_size, actor_params_num, tau, lr, gamma):
        self.sess =sess
        self.screen_size = screen_size
        self.tau = tau
        self.lr = lr
        self.gamma = gamma

        with tf.variable_scope('critic') :
            self.inputs, self.out = self.create_critic_network()

        self.critic_network_params = tf.trainable_variables()[actor_params_num:]

        # Target critic network 생성
        with tf.variable_scope('target_critic_network'):
            self.target_inputs, self.target_out = self.create_critic_network()

        self.target_critic_network_params = tf.trainable_variables()[
                                            (len(self.critic_network_params) + actor_params_num):]

        self.update_target_critic_network_params = \
            [self.target_critic_network_params[i].assign(tf.multiply(self.critic_network_params[i], self.tau) \
                                                  + tf.multiply(self.target_critic_network_params[i], 1. - self.tau))
             for i in range(len(self.target_critic_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.out))
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

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

    def train(self, s, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs : s,
            self.predicted_q_value : predicted_q_value
        })

    def predict(self, s):

        #s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        q = self.sess.run(self.out, feed_dict={
            self.inputs: s
        })
        return q

    def target_predict(self, s):
        #s = s[:,:,:,np.newaxis]
        #s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        q = self.sess.run(self.target_out, feed_dict={
            self.target_inputs: s
        })
        return q

    def update_target_critic_network(self):
        self.sess.run(self.update_target_critic_network_params)