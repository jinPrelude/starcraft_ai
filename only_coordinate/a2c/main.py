import argparse
from collections import deque
from absl import flags
import sys

import numpy as np
import tensorflow as tf

from pysc2.env import sc2_env
from pysc2.lib import actions
from only_coordinate.a2c.agent import actorNetwork, criticNetwork

def build_summaries() :
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def train(sess, env, actor, critic, args) :
    summary_ops, summary_vars = build_summaries()

    saver = tf.train.Saver()

    if args['load_model'] :
        saver.restore(sess, args['saved_model_directory'])
    else :
        sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    state_stack = deque(maxlen=4)
    reward_mean = deque(maxlen=10)

    for episode in range(args['max_episode']) :

        episode_reward = 0
        episode_max_q = 0

        state = env.reset()
        available_action = state[0].observation.available_actions
        available_action = state[0].observation.available_actions
        state = state[0].observation.feature_screen[5]

        for _ in range(state_stack.maxlen) :
            state_stack.append(state)

        for step in range(args['max_episode_step']) :

            state_stack_arr = np.asarray(state_stack)
            state_stack_arr = np.reshape(state_stack_arr, (1, 64, 64, 4))
            x, y, select = actor.predict(state_stack_arr)
            if args['map_name'] == 'DefeatRoaches' :
                if actions.FUNCTIONS.Attack_screen.id in available_action :
                    a = [actions.FUNCTIONS.Attack_screen("now", (x, y))]
                else :
                    a = [actions.FUNCTIONS.no_op()]
            elif args['map_name'] == 'CollectMineralShards' :
                if actions.FUNCTIONS.select_point.id in available_action :
                    a = [actions.FUNCTIONS.select_point("select", (x, y))]
                else :
                    a = [actions.FUNCTIONS.no_op()]

            state2 = env.step(a)

            r = state2[0].reward
            terminal = state2[0].last()
            state2 = state2[0].observation.feature_screen[5]

            state2_stack = state_stack
            state2_stack.append(state2)
            state2_stack_arr = np.asarray(state2_stack)
            state2_stack_arr = np.reshape(state_stack_arr, (1, 64, 64, 4))
            predicted_q = critic.predict(state_stack_arr)
            target_value = r + critic.gamma * critic.predict(state2_stack_arr)
            critic.train(target_value, state_stack_arr)

            advantage = (r + critic.gamma * critic.predict(state2_stack_arr)) - predicted_q
            actor.train(state_stack_arr, advantage, select[0])

            state_stack = state2_stack
            episode_reward += r
            episode_max_q += predicted_q[0][0]

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: episode_reward,
                    summary_vars[1]: episode_max_q / float(step)
                })

                writer.add_summary(summary_str, episode)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(episode_reward),
                                                                             episode, (episode_max_q / float(step))))
                reward_mean.append(episode_reward)

                break

        if episode > 10 :
            reward_reduce_mean = int(sum(reward_mean)/len(reward_mean))

            if reward_reduce_mean > 30 :
                    saver.save(sess, args['summary_dir'])
                    break


def main(args) :
    with tf.Session() as sess :

        with sc2_env.SC2Env(
            map_name=args['map_name'],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(
                    screen=args['screen_size'],
                    minimap=args['minimap_size']
                )
            ),
            step_mul=args['step_mul'],
            game_steps_per_episode=args['max_episode_step'],
            visualize=True
        ) as env :
            actor = actorNetwork(sess, args['actor_lr'], args['screen_size'], args['action_size'], )
            critic = criticNetwork(sess, args['critic_lr'], args['screen_size'], args['gamma'])

            train(sess, env, actor, critic, args)

if __name__=="__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('--map_name', default='DefeatRoaches')
    parser.add_argument('--step_mul', default=8)
    parser.add_argument('--screen_size', default=64)
    parser.add_argument('--minimap_size', default=16)
    parser.add_argument('--max_episode_step', default=300)
    parser.add_argument('--max_episode', default=10000)

    parser.add_argument('--load_model', default=False)
    parser.add_argument('--saved_model_directory', default='./results/save_model')
    parser.add_argument('--summary_dir', default='./results/tensorboard')

    parser.add_argument('--actor_lr', default=0.001)
    parser.add_argument('--critic_lr', default=0.01)
    parser.add_argument('--action_size', default=1)
    parser.add_argument('--gamma', default=0.99)

    flags.FLAGS(sys.argv)
    
    args = vars(parser.parse_args())
    main(args)