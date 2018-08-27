from pysc2.lib import actions
import tensorflow as tf
import numpy as np
from only_DDPG.OU_Noise import OrnsteinUhlenbeckActionNoise

def postprocessing(s, x, y, available_action) :

    s = s[:,:,:,-1]
    dim = np.shape(s)[0]
    # 1개 들어올때만 생각했고 batch로 들어올때 생각을 못함

    y_i = []
    for i in range(dim) :
        action = actions.FUNCTIONS.no_op()
        if x != 0 and y != 0:
            if s[i][x][y] == 0:
                if actions.FUNCTIONS.Move_screen.id in available_action :
                    action = actions.FUNCTIONS.Move_screen("now", (x, y))

            elif s[i][x][y] == 1:
                if actions.FUNCTIONS.select_point.id in available_action :
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

def act(out_descrete, available_action, coordinate) :
    dim = np.shape(out_descrete)[0]


    y_i = []
    for i in range(dim):
        #print('out_descrete[%d] : '%i, out_descrete[i])
        action = actions.FUNCTIONS.no_op()
        if out_descrete[i] == 0:
            if actions.FUNCTIONS.Move_screen.id in available_action:
                action = actions.FUNCTIONS.Move_screen("now", (coordinate[i][0], coordinate[i][1]))

        elif out_descrete[i] == 1:
            if actions.FUNCTIONS.select_point.id in available_action:
                action = actions.FUNCTIONS.select_point("select", (coordinate[i][0], coordinate[i][1]))

        elif out_descrete[i] == 2:
            if actions.FUNCTIONS.Attack_screen.id in available_action:
                action = actions.FUNCTIONS.Attack_screen("now", (coordinate[i][0], coordinate[i][1]))
        else :
            if actions.FUNCTIONS.Attack_screen.id in available_action:
                action = actions.FUNCTIONS.Attack_screen("now", (coordinate[i][0], coordinate[i][1]))
                #print('attack', coordinate[i][0], coordinate[i][1])
            else :
                actions.FUNCTIONS.no_op()
                #print('else', coordinate[i][0], coordinate[i][1])

        y_i.append(action)
    # print('end')
    #print('y_i : ', y_i)
    if dim == 1:
        return y_i[0]
    else:
        return y_i


class actorNetwork() :
    def __init__(self, sess, screen_size, action_dim, action_bound, tau, batch_size, lr):
        self.sess = sess
        self.screen_size = screen_size
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.tau = tau
        self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.array([5., 5.]), sigma=5)
        self.descrete_action_dim = 4
        self.descrete_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.descrete_action_dim), sigma=0.5)
        self.batch_size = batch_size
        self.lr = lr
        self.optimize_switch = True

        with tf.variable_scope('actor') :
            self.inputs, self.out, self.out_descrete = self.create_actor_network()

        self.actor_network_params = tf.trainable_variables()

        with tf.variable_scope('target_actor') :
            self.target_inputs, self.target_out, self.target_out_descrete = self.create_actor_network()

        self.target_actor_network_params = tf.trainable_variables()[len(self.actor_network_params):]

        self.trainable_params_num = len(self.actor_network_params) + len(self.target_actor_network_params)

        self.update_target_actor_network_params = \
            [self.target_actor_network_params[i].assign(tf.multiply(self.actor_network_params[i], self.tau) +
                                                  tf.multiply(self.target_actor_network_params[i], 1. - self.tau)) for i in
             range(len(self.target_actor_network_params))]


        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])
        self.descrete_action_gradient = tf.placeholder(tf.float32, [None, self.descrete_action_dim])

        self.unnormalized_actor_continuous_gradients = tf.gradients(
            self.out, self.actor_network_params, -self.action_gradient)[:8]

        self.actor_network_params_descrete = self.actor_network_params[:6] + self.actor_network_params[8:]

        self.unnormalized_actor_descrete_gradients = tf.gradients(
            self.out_descrete, self.actor_network_params_descrete, -self.descrete_action_gradient)

        self.unnormalized_actor_gradients = self.unnormalized_actor_continuous_gradients + self.unnormalized_actor_descrete_gradients

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        self.optimize_continuous = \
            tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients[:8], self.actor_network_params))

        self.optimize_descrete = \
            tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients[8:], self.actor_network_params_descrete))



    def create_actor_network(self):
        inputs = tf.placeholder(tf.float32, shape=[None, self.screen_size, self.screen_size, 4])
        init = tf.contrib.layers.xavier_initializer()
        conv1 = tf.layers.conv2d(inputs=inputs, filters=8, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.tanh)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.tanh)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.tanh)

        conv_flat = tf.layers.flatten(conv3)

        w1 = tf.get_variable(name='w1', shape=[64 * 64, 100], dtype=tf.float32,
                             initializer=init)
        l1 = tf.matmul(conv_flat, w1)
        l1 = tf.nn.tanh(l1)

        w2 = tf.get_variable(name='w2', shape=[100, 2], dtype=tf.float32,
                             initializer=init)
        l2 = tf.matmul(l1, w2)
        out = tf.nn.tanh(l2)
        out = tf.multiply(out, self.action_bound)
        out = tf.add(out, 32.)

        w2_descrete = tf.get_variable(name='w2_descrete', shape=[100, 50], dtype=tf.float32,
                             initializer=init)
        l2_descrete = tf.matmul(l1, w2_descrete)
        l2_descrete = tf.nn.tanh(l2_descrete)

        w3_descrete = tf.get_variable(name='w3_descrete', shape=[50, 4], dtype=tf.float32,
                                      initializer=init)
        l3_descrete = tf.matmul(l2_descrete, w3_descrete)
        l3_descrete = tf.nn.softmax(l3_descrete)
        #l3_descrete = tf.argmax(l3_descrete, axis=1)


        return inputs, out, l3_descrete

    def train(self, inputs, a_gradient, a_d_gradient):
        if self.optimize_switch is True :
            self.sess.run(self.optimize_continuous, feed_dict={
                self.inputs: inputs,
                self.action_gradient: a_gradient,
                self.descrete_action_gradient : a_d_gradient
            })
            self.optimize_switch = False

        else :
            self.sess.run(self.optimize_descrete, feed_dict={
                self.inputs: inputs,
                self.action_gradient: a_gradient,
                self.descrete_action_gradient: a_d_gradient
            })
            self.optimize_switch = True

    def predict(self, s, available_action, random):
        #s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        out, out_descrete = self.sess.run([self.out, self.out_descrete], feed_dict={
            self.inputs : s
        })
        out[0] += self.action_noise()
        #out_descrete += self.descrete_action_noise() #
        out_descrete = np.argmax(out_descrete, axis=1) #
        one_hot = np.zeros((np.shape(out_descrete)[0],4))
        for t in range(np.shape(one_hot)[0]) :
            one_hot[t][out_descrete[t]] = 1
        out = out.astype(int)
        out = np.clip(out, 1, 63)
        """
        for k in range(out.shape[0]) :
            outtf.clip_by_value(out[k][0], 1, 63)
            tf.clip_by_value(out[k][1], 1, 63)
            """
        #action = postprocessing(s, out[0][0], out[0][1], available_action)
        if random :
            out = np.random.uniform(10, 53, (1, 2))
            out = out.astype(int)

            out_descrete = np.random.randint(0, 3, (1))  #
            one_hot = np.zeros((np.shape(out_descrete)[0], 4))
            for t in range(np.shape(one_hot)[0]):
                one_hot[t][out_descrete[t]] = 1

        action = act(out_descrete, available_action, out)

        return action, out, one_hot

    def target_predict(self, s, available_action):
        #s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        out, out_descrete = self.sess.run([self.target_out, self.target_out_descrete], feed_dict={
            self.target_inputs : s,
        })
        # 좌표값이 오지게 64랑 1만 나오낟
        out_descrete = np.argmax(out_descrete, axis=1) ## 타입 이상함
        one_hot = np.zeros((np.shape(out_descrete)[0],4))
        for t in range(np.shape(one_hot)[0]) :
            one_hot[t][out_descrete[t]] = 1
        out = out.astype(int)
        out = np.clip(out, 1, 63)
        action = act(out_descrete, available_action, out)

        return action, out, one_hot

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
        self.descrete_action_dim = 4

        with tf.variable_scope('critic') :
            self.inputs, self.actions, self.descrete_actions, self.out = self.create_critic_network()

        self.critic_network_params = tf.trainable_variables()[actor_params_num:]

        # Target critic network 생성
        with tf.variable_scope('target_critic_network'):
            self.target_inputs, self.target_actions, self.target_descrete_actions, self.target_out = self.create_critic_network()

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

        self.descrete_q_grads = tf.gradients(self.out, self.descrete_actions)

    def create_critic_network(self):
        inputs = tf.placeholder(tf.float32, shape=[None, self.screen_size, self.screen_size, 4])
        actions = tf.placeholder(tf.float32, shape=[None, self.action_dim])
        descrete_actions = tf.placeholder(tf.float32, shape=[None, self.descrete_action_dim])

        conv1 = tf.layers.conv2d(inputs=inputs, filters=8, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.tanh)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.tanh)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.tanh)

        conv_flat = tf.layers.flatten(conv3)
        init = tf.contrib.layers.xavier_initializer()
        w1 = tf.get_variable(name='w1', shape=[64 * 64, 100], dtype=tf.float32, initializer=init)
        l1 = tf.matmul(conv_flat, w1)
        l1 = tf.nn.tanh(l1)

        w2 = tf.get_variable(name='w2', shape=[100, 50], dtype=tf.float32, initializer=init)
        l2_tmp = tf.matmul(l1, w2)

        w1_a = tf.get_variable(name='w1_a', shape=[2, 50], dtype=tf.float32, initializer=init)
        l1_a = tf.matmul(actions, w1_a)

        w1_a_d = tf.get_variable(name='w1_a_d', shape=[4, 50], dtype=tf.float32, initializer=init)
        l1_a_d = tf.matmul(descrete_actions, w1_a_d)

        l2 = tf.add(l2_tmp, l1_a)
        l2 = tf.add(l1_a, l1_a_d)
        l2 = tf.nn.tanh(l2)

        w3 = tf.get_variable(name='w3', shape=[50, 1], dtype=tf.float32, initializer=init)
        out = tf.matmul(l2, w3)

        return inputs, actions, descrete_actions, out

    def train(self, s, predicted_q_value, action, descrete_action):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs : s,
            self.predicted_q_value : predicted_q_value,
            self.actions : action,
            self.descrete_actions : descrete_action
        })

    def predict(self, s, action, descrete_action):

        #s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        action = np.reshape(action, (-1, 2))
        q = self.sess.run(self.out, feed_dict={
            self.inputs: s,
            self.actions : action,
            self.descrete_actions : descrete_action
        })
        return q

    def target_predict(self, s, action, descrete_action):
        #s = s[:,:,:,np.newaxis]
        #s = np.reshape(s, (-1, self.screen_size, self.screen_size, 4))
        action = np.reshape(action, (-1, 2))
        q = self.sess.run(self.target_out, feed_dict={
            self.target_inputs: s,
            self.target_actions : action,
            self.target_descrete_actions : descrete_action
        })
        return q

    def q_gradient(self, s, a, a_d):
        return self.sess.run(self.q_grads, feed_dict={
            self.inputs: s,
            self.actions: a,
            self.descrete_actions :  a_d
        })

    def descrete_q_gradient(self, s, a, a_d):
        return self.sess.run(self.descrete_q_grads, feed_dict={
            self.inputs : s,
            self.actions : a,
            self.descrete_actions : a_d
        })

    def update_target_critic_network(self):
        self.sess.run(self.update_target_critic_network_params)