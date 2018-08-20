import argparse
import tensorflow as tf
import numpy as np
from pysc2.lib import actions
from collections import deque
from pysc2.env import sc2_env
from agent import actorNetwork, criticNetwork
from absl import flags
import sys
from replay_buffer import ReplayBuffer

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def is_available(available_action, a) :


    #s = state[:,:,:,-1]
    dim = len(a)
    # 1개 들어올때만 생각했고 batch로 들어올때 생각을 못함

    y_i = []
    for i in range(dim) :

        if not (a.id in available_action) :
            result = actions.FUNCTIONS.no_op()
        else :
            result = a
        y_i.append(result)

    #print('end')
    print('is_available function work')
    if dim == 1 :
        return y_i
    else :
        return y_i

    """
    if not (a in state.observation.available_actions) :
        result = actions.FUNCTIONS.no_op()
    else :
        result = a

    print('is_available function work')
    return result
    """

def train(sess, env, actor, critic, args, replay_buffer) :

    summary_ops, summary_vars = build_summaries()

    saver = tf.train.Saver()

    if args['load_model'] :
        saver.restore(sess, args['saved_model_directory'])
    else :
        sess.run(tf.global_variables_initializer())

    #generate tensorboard
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)


    state_stack = deque(maxlen=4)
    reward_mean = deque(maxlen=10)

    for episode in range(args['episode']):

        episode_reward = 0
        episode_max_q = 0

        # reset state
        state = env.reset()

        # extract player_relative layers
        available_action = state[0].observation.available_actions
        state = state[0].observation.feature_screen[5]

        # stack initial state
        for _ in range(state_stack.maxlen) :
            state_stack.append(state)


        for step in range(args['max_episode_step']):

            state_stack_arr = np.asarray(state_stack) # Change type to save relay_buffer and treat easily, shape=(4, 64, 64)
            state_stack_arr = np.reshape(state_stack_arr, (-1, args['screen_size'], args['screen_size'], 4))
            a, a_raw = actor.predict(state_stack_arr, available_action)
            a = [a]

            state2 = env.step(a)
            available_action = state2[0].observation.available_actions

            # Add to replay buffer
            r = state2[0].reward
            terminal = state2[0].last()
            state2 = state2[0].observation.feature_screen[5]

            # Generate state2_stack to save in replay_buffer
            state2_stack = state_stack
            state2_stack.append(state2)
            state2_stack_arr = np.asarray(state2_stack)
            state2_stack_arr = np.reshape(state2_stack_arr, (-1, args['screen_size'], args['screen_size'], 4))

            replay_buffer.add(state_stack_arr, a_raw[0], r, terminal, state2_stack_arr)

            #print('state_stack shape : ', np.shape(state_stack_arr), '  | reward : ', r, '  action : ', a_raw,
            #      ' | predicted_q : ', critic.predict(state_stack_arr, a_raw))
            #input()

            if replay_buffer.size() > args['train_start'] :
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                s_batch = np.reshape(s_batch, (-1, args['screen_size'], args['screen_size'], 4))
                s2_batch = np.reshape(s2_batch, (-1, args['screen_size'], args['screen_size'], 4))
                _, target_a_batch = actor.target_predict(s2_batch, available_action)
                target_q = critic.target_predict(s2_batch, target_a_batch)

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                y_i = np.asarray(y_i)
                y_i = np.reshape(y_i, (args['minibatch_size'], 1))
                predicted_q_value, _ = critic.train(s_batch, y_i, a_batch)

                episode_max_q += np.amax(predicted_q_value)

                _, action = actor.predict(s_batch, available_action)

                #test = actor.q_gradients(predicted_q_value, action)
                grads = critic.q_gradient(s_batch, action)
                actor.train(s_batch, grads[0])

                actor.update_target_actor_network()
                critic.update_target_critic_network()

            state_stack = state2_stack
            episode_reward += r

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

def main(args) :
    with tf.Session() as sess :


        with sc2_env.SC2Env(
            map_name=args['map_name'],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=args['screen_size'],
                                                      minimap=args['minimap_size'])
            ),
            step_mul=args['step_mul'],
            game_steps_per_episode=args['max_episode_step'],
            visualize=True
        ) as env :
            action_bound = int(args['screen_size']) / int(2)
            # sess, screen_size, action_dim, learning_rate, action_bound, minibatch_size, tau
            actor = actorNetwork(sess, args['screen_size'], args['action_dim'], action_bound,
                                 args['tau'], args['minibatch_size'], args['actor_lr'])

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
    parser.add_argument('--step_mul', default=8)
    parser.add_argument('--max_episode_step', default=500)
    parser.add_argument('--episode', default=100)
    parser.add_argument('--tau', default=0.01)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--buffer_size', default=100000)
    parser.add_argument('--minibatch_size', default=16)
    parser.add_argument('--load_model', default=False)
    parser.add_argument('-saved_model_directory', default='./results/save_model')
    parser.add_argument('--summary_dir', default='./results/tensorboard')
    parser.add_argument('--train_start', default=100)

    parser.add_argument('--actor_lr', default=0.001)
    parser.add_argument('--critic_lr', default=0.01)

    flags.FLAGS(sys.argv)

    args = vars(parser.parse_args())

    main(args)