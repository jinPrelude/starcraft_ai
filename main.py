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

def train(sess, env, actor, critic, args, replay_buffer) :

    sess.run(tf.global_variables_initializer())

    state_stack = deque(maxlen=4)

    for episode in range(args['episode']):

        episode_reward = 0
        episode_max_q = 0

        # reset state
        state = env.reset()

        # extract player_relative layers
        state = state[0].observation.feature_screen[5]

        # stack initial state
        for _ in range(state_stack.maxlen) :
            state_stack.append(state)


        for step in range(args['max_episode_step']):

            state_stack_arr = np.asarray(state_stack) # Change type to save relay_buffer and treat easily, shape=(4, 64, 64)
            state_stack_arr = np.reshape(state_stack_arr, (-1, args['screen_size'], args['screen_size'], 4))
            a, a_raw = actor.predict(state_stack_arr)
            a = [a]
            a_raw = np.reshape(a_raw, (1, args['action_dim']))
            state2 = env.step(a)


            # Add to replay buffer
            r = state2[0].reward
            terminal = state2[0].last()
            state2 = state2[0].observation.feature_screen[5]

            # Generate state2_stack to save in replay_buffer
            state2_stack = state_stack
            state2_stack.append(state2)
            state2_stack_arr = np.asarray(state2_stack)


            replay_buffer.add(state_stack_arr, a_raw[0], r, terminal, state2_stack_arr)

            print('state_stack shape : ', np.shape(state_stack_arr), '  | reward : ', r, '  action : ', a_raw,
                  ' | predicted_q : ', critic.predict(state_stack_arr, a_raw))
            #input()

            if replay_buffer.size() > args['minibatch_size'] :
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                s_batch = np.reshape(s_batch, (-1, args['screen_size'], args['screen_size'], 4))
                s2_batch = np.reshape(s2_batch, (-1, args['screen_size'], args['screen_size'], 4))
                target_q = critic.target_predict(s2_batch, a_batch)

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                predicted_q_value, _ = critic.train(
                    s_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)), a_raw)

                episode_max_q += np.amax(predicted_q_value)

                _, action = actor.predict(s_batch)

                #test = actor.q_gradients(predicted_q_value, action)
                test = critic.q_gradient(s_batch, action)
                print(test)

            state_stack = state2_stack

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
            critic = criticNetwork(sess, args['screen_size'], actor.get_trainable_params_num(),
                                   args['tau'], args['critic_lr'], args['gamma'], args['action_dim'])

            replay_buffer = ReplayBuffer(buffer_size=args['buffer_size'])

            train(sess, env, actor, critic, args, replay_buffer)



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