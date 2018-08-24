from pysc2.lib import actions
import tensorflow as tf
import numpy as np
from OU_Noise import OrnsteinUhlenbeckActionNoise

def postprocessing(s, coordinate, available_action) :

    s = s[:,:,:,-1]
    dim = np.shape(s)[0]
    # 1개 들어올때만 생각했고 batch로 들어올때 생각을 못함

    y_i = []
    x = 0
    y = 0
    for i in range(dim) :
        x = coordinate[i][0]
        y = coordinate[i][1]
        action = actions.FUNCTIONS.no_op()
        if (x > np.shape(s)[1] - 1) or (y > np.shape(s)[1] - 1) or (x < 1) or (y < 1):
            action = actions.FUNCTIONS.no_op()
            return action
        else :
            if x != 0 and y != 0:
                if s[i][x][y] == 0:
                    if actions.FUNCTIONS.Attack_screen.id in available_action :
                        action = actions.FUNCTIONS.Attack_screen("now", (x, y))

                elif s[i][x][y] == 1:
                    if actions.FUNCTIONS.Attack_screen.id in available_action :
                        action = actions.FUNCTIONS.select_point("select", (x, y))

                elif s[i][x][y] == 4:
                    if actions.FUNCTIONS.Attack_screen.id in available_action :
                        action = actions.FUNCTIONS.Attack_screen("now", (x, y))
            y_i.append(action)
        #print('end')
    if dim == 1 :
        return y_i[0]
    else :
        return y_i


class actorNetwork() :
    def __init__(self, sess, screen_size, action_dim, action_bound, tau, batch_size, lr):
        self.sess = sess
        self.screen_size = screen_size
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.tau = tau
        self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.array([5., 5.]), sigma=2)
        self.batch_size = batch_size
        self.lr = lr

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

        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])

        self.unnormalized_actor_gradients = tf.gradients(
            self.out, self.actor_network_params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        self.optimize = \
            tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.actor_network_params))



    def create_actor_network(self):
        inputs = tf.placeholder(tf.float32, shape=[None, self.screen_size, self.screen_size, 4])
        init = tf.random_uniform_initializer(-0.05, 0.05)
        conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.leaky_relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.leaky_relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.leaky_relu)

        conv_flat = tf.layers.flatten(conv3)

        w1 = tf.get_variable(name='w1', shape=[64 * 64, 100], dtype=tf.float32,
                             initializer=init)
        l1 = tf.matmul(conv_flat, w1)
        l1 = tf.nn.leaky_relu(l1)

        w2 = tf.get_variable(name='w2', shape=[100, 2], dtype=tf.float32,
                             initializer=init)
        l2 = tf.matmul(l1, w2)
        l2 = tf.nn.leaky_relu(l2)
        #out2 = tf.multiply(out3, self.action_bound)
        #out = tf.add(out2, 32.)

        return inputs, l2
    """
    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })
    """
    def train(self, inputs, a_gradient):
        _, target_actor_network_params, ummormalized_actor_gradients , actor_gradients = \
            self.sess.run([self.optimize, self.target_actor_network_params, self.unnormalized_actor_gradients,
                       self.actor_gradients], feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })
        print('bebebebe')

    def predict(self, s, available_action, random):
        #s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        out = self.sess.run(self.out, feed_dict={
            self.inputs : s
        })
        out[0] += self.action_noise()
        print()
        print(out)
        out = np.asarray(out)
        #out = out/10.
        out = out.astype(int)
        """
        for k in range(out.shape[0]) :
            outtf.clip_by_value(out[k][0], 1, 63)
            tf.clip_by_value(out[k][1], 1, 63)
            """

        if random :
            out = np.random.uniform(10, 53, (1, 2))
            out = out.astype(int)

            out_descrete = np.random.randint(0, 3, (1))  #
            one_hot = np.zeros((np.shape(out_descrete)[0], 4))
            for t in range(np.shape(one_hot)[0]):
                one_hot[t][out_descrete[t]] = 1

        action = postprocessing(s, out, available_action)




        return action, out

    def target_predict(self, s, available_action):
        #s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        out = self.sess.run(self.target_out, feed_dict={
            self.target_inputs : s,
        })
        out[0] -= self.action_noise()
        out = np.asarray(out)
        out = out.astype(int)
        out = np.clip(out, 1, 63)
        action = postprocessing(s, out, available_action)

        return action, out

    def update_target_actor_network(self):
        self.sess.run(self.update_target_actor_network_params)

    def get_trainable_params_num(self):
        return self.trainable_params_num

###############################################################

class criticNetwork(object):

    def __init__(self, sess, screen_size, actor_params_num, tau, lr, gamma, action_dim):
        self.sess =sess
        self.screen_size = screen_size
        self.tau = tau
        self.lr = lr
        self.gamma = gamma
        self.action_dim = action_dim

        with tf.variable_scope('critic') :
            self.inputs, self.actions, self.out = self.create_critic_network()

        self.critic_network_params = tf.trainable_variables()[actor_params_num:]

        # Target critic network 생성
        with tf.variable_scope('target_critic_network'):
            self.target_inputs, self.target_actions, self.target_out = self.create_critic_network()

        self.target_critic_network_params = tf.trainable_variables()[
                                            (len(self.critic_network_params) + actor_params_num):]

        self.update_target_critic_network_params = \
            [self.target_critic_network_params[i].assign(tf.multiply(self.critic_network_params[i], self.tau) \
                                                  + tf.multiply(self.target_critic_network_params[i], 1. - self.tau))
             for i in range(len(self.target_critic_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.out))
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.q_grads = tf.gradients(self.out, self.actions)

    def create_critic_network(self):
        inputs = tf.placeholder(tf.float32, shape=[None, self.screen_size, self.screen_size, 4])
        actions = tf.placeholder(tf.float32, shape=[None, self.action_dim])
        conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.leaky_relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.leaky_relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.leaky_relu)

        conv_flat = tf.layers.flatten(conv3)
        init = tf.random_uniform_initializer(-0.3, 0.3)
        w1 = tf.get_variable(name='w1', shape=[64 * 64, 100], dtype=tf.float32, initializer=init)
        l1 = tf.matmul(conv_flat, w1)
        l1 = tf.nn.leaky_relu(l1)

        w2 = tf.get_variable(name='w2', shape=[100, 50], dtype=tf.float32, initializer=init)
        l2_tmp = tf.matmul(l1, w2)

        w1_a = tf.get_variable(name='w1_a', shape=[2, 50], dtype=tf.float32, initializer=init)
        l1_a = tf.matmul(actions, w1_a)

        l2 = tf.add(l2_tmp, l1_a)
        l2 = tf.nn.leaky_relu(l2)

        w3 = tf.get_variable(name='w3', shape=[50, 1], dtype=tf.float32, initializer=init)
        out = tf.matmul(l2, w3)

        return inputs, actions, out

    def train(self, s, predicted_q_value, action):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs : s,
            self.predicted_q_value : predicted_q_value,
            self.actions : action
        })

    def predict(self, s, action):

        #s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        q = self.sess.run(self.out, feed_dict={
            self.inputs: s,
            self.actions : action
        })
        return q

    def target_predict(self, s, action):
        #s = s[:,:,:,np.newaxis]
        #s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        q = self.sess.run(self.target_out, feed_dict={
            self.target_inputs: s,
            self.target_actions : action
        })
        return q

    def q_gradient(self, s, a):
        return self.sess.run(self.q_grads, feed_dict={
            self.inputs: s,
            self.actions: a
        })

    def update_target_critic_network(self):
        self.sess.run(self.update_target_critic_network_params)