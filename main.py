import argparse
import tensorflow as tf
import numpy as np
from pysc2.lib import actions
from collections import deque
from pysc2.env import sc2_env
from agent import actorNetwork, criticNetwork
from absl import flags
import sys
from OU_Noise import OrnsteinUhlenbeckActionNoise
from replay_buffer import ReplayBuffer

def train(sess, env, actor, critic, args, action_noise, replay_buffer) :

    sess.run(tf.global_variables_initializer())

    state_stack = deque(maxlen=4)

    for episode in range(args['episode']):

        # episode_reward = 0
        # episode_max_q = 0

        # reset state
        s = env.reset()

        # extract player_relative layers
        s = s[0].observation.feature_screen[5]

        # stack initial state
        for _ in range(state_stack.maxlen) :
            state_stack.append(s)


        for step in range(args['max_episode_step']):

            state_stack_arr = np.asarray(state_stack) # Change type to save relay_buffer and treat easily, shape=(4, 64, 64)
            a = [actor.predict(state_stack_arr)]
            s2 = env.step(a)


            # Add to replay buffer

            r = s2[0].reward
            terminal = s2[0].last()
            s2 = s2[0].observation.feature_screen[5]

            replay_buffer.add(state_stack_arr, a, r, terminal, s2)

            print('state_stack shape : ', np.shape(state_stack_arr), '  | reward : ', r, '  action : ', a[0].arguments,
                  ' | predicted_q : ', critic.predict(state_stack_arr))
            #input()

            if replay_buffer.size() > args['minibatch_size'] :
                

            state_stack.append(s2)

def main(args) :
    with tf.Session() as sess :


        with sc2_env.SC2Env(
            map_name=args['map_name'],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=args['screen_size'],
                                                      minimap=args['minimap_size'])
            ),
            step_mul=args['step_mul'],
            game_steps_per_episode=args['max_episode_step']
        ) as env :
            action_bound = int(args['screen_size']) / int(2) - 1
            # sess, screen_size, action_dim, learning_rate, action_bound, minibatch_size, tau
            actor = actorNetwork(sess, args['screen_size'], args['action_dim'], action_bound,
                                 args['tau'])

            # sess, screen_size, action_dim, learning_rate, tau, gamma, num_actor_vars, minibatch_size
            critic = criticNetwork(sess, args['screen_size'], actor.trainable_params_num(),
                                   args['tau'])

            replay_buffer = ReplayBuffer(buffer_size=args['buffer_size'])
            action_noise = OrnsteinUhlenbeckActionNoise(mu=0.)
            train(sess, env, actor, critic, args, action_noise, replay_buffer)



if __name__=="__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('--map_name', default='DefeatRoaches')
    parser.add_argument('--screen_size', default=64)
    parser.add_argument('--action_dim', default=2)
    parser.add_argument('--minimap_size', default=64)
    parser.add_argument('--step_mul', default=1)
    parser.add_argument('--max_episode_step', default=500)
    parser.add_argument('--episode', default=100)
    parser.add_argument('--tau', default=0.01)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--buffer_size', default=100000)
    parser.add_argument('--minibatch_size', default=64)

    parser.add_argument('--actor_lr', default=0.001)
    parser.add_argument('--critic_lr', default=0.01)

    flags.FLAGS(sys.argv)

    args = vars(parser.parse_args())

    main(args)