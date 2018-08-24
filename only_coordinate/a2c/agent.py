import tensorflow as tf
import numpy as np
from pysc2.lib import actions

class actorNetwork() :
    def __init__(self, sess, lr, screen_size, action_size):
        self.sess = sess
        self.lr = lr
        self.action_size = action_size
        self.screen_size = screen_size
        with tf.variable_scope('actor') :
            self.inputs, self.out = self.create_actor_network()

        self.advantage = tf.placeholder(tf.float32, shape=None)
        self.select = tf.placeholder(tf.int32, shape=None)
        self.action_prob = self.out[0,self.select]
        self.action_prob = tf.reshape(self.action_prob, [1])
        self.loss = -tf.log(self.action_prob) * self.advantage
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def create_actor_network(self):
        inputs = tf.placeholder(tf.float32, shape=[1, 64, 64, 4])

        conv1 = tf.layers.conv2d(inputs=inputs, filters=4, kernel_size=[3, 3],
                                 padding='same', activation=tf.nn.leaky_relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=4, kernel_size=[3, 3],
                                 padding='same', activation=tf.nn.leaky_relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=4, kernel_size=[3, 3],
                                 padding='same', activation=tf.nn.leaky_relu)

        conv4 = tf.layers.conv2d(inputs=conv3, filters=1, kernel_size=[3, 3],
                                 padding='same', activation=tf.nn.leaky_relu)
        conv4 = tf.layers.flatten(conv4)
        out = tf.nn.softmax(conv4)

        return inputs, out

    def out_preprocess(self, out):
        out = np.reshape(out, (64*64))
        grid = np.arange(0, 64*64)
        select = np.random.choice(grid, 1, p=out)
        x = int(select/self.screen_size)
        y = int(select % self.screen_size)

        return x, y, select

    def train(self, obs, advantage, select):
        self.sess.run(self.optimizer, feed_dict={
            self.inputs : obs,
            self.advantage : advantage,
            self.select : select
        })

    def predict(self, obs):
        out = self.sess.run(self.out, feed_dict={
            self.inputs : obs
        })
        x, y, select = self.out_preprocess(out)

        return x, y, select



################### Critic Network ########################

class criticNetwork() :
    def __init__(self, sess, lr, screen_size, gamma):
        self.sess = sess
        self.lr = lr
        self.screen_size = screen_size
        self.gamma = gamma
        with tf.variable_scope('critic') :
            self.inputs, self.out = self.create_critic_network()

        self.target_value = tf.placeholder(tf.float32, shape=None)

        loss = tf.square(self.target_value - self.out)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)


    def create_critic_network(self):
        inputs = tf.placeholder(tf.float32, shape=[1, 64, 64, 4])

        init = tf.random_uniform_initializer(-1., 1.)
        conv1 = tf.layers.conv2d(inputs=inputs, filters=4, kernel_size=[3, 3],
                                 padding='same', activation=tf.nn.leaky_relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=4, kernel_size=[3, 3],
                                 padding='same', activation=tf.nn.leaky_relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=[3, 3],
                                 padding='same', activation=tf.nn.leaky_relu)
        conv_flat = tf.layers.flatten(conv3)
        w1 = tf.get_variable(name='critic_w1', shape=[64*64, 100], initializer=init)
        l1 = tf.matmul(conv_flat, w1)
        l1 = tf.nn.leaky_relu(l1)

        w2 = tf.get_variable(name='critic_w2', shape=[100, 50], initializer=init)
        l2 = tf.matmul(l1, w2)
        l2 = tf.nn.leaky_relu(l2)

        w3 = tf.get_variable(name='critic_w3', shape=[50, 1], initializer=init)
        out = tf.matmul(l2, w3)

        return inputs, out




    def train(self, target_value, obs):

        self.sess.run(self.optimizer, feed_dict={
            self.target_value : target_value,
            self.inputs : obs
        })

    def predict(self, obs):

        return self.sess.run(self.out, feed_dict={
            self.inputs : obs
        })