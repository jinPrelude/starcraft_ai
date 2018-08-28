import argparse
import tensorflow as tf
import numpy as np
from collections import deque
from pysc2.env import sc2_env
from pysc2.lib import actions
from DPG_SPG.agent import actorNetwork, criticNetwork, actorNetwork_SPG
from absl import flags
import sys
from DPG_SPG.replay_buffer import ReplayBuffer

def act(out_descrete, available_action, coordinate) :
    dim = out_descrete.size


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


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


def train(sess, env, actor, actor_SPG, critic, args, replay_buffer) :

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
            a_coordinate = actor.predict(state_stack_arr, False)
            a_select, a_select_prob = actor_SPG.predict(state_stack_arr)
            #print(a_raw)
            #타입 맞춰주기용
            a = act(a_select, available_action, a_coordinate)
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

            replay_buffer.add(state_stack_arr, a_coordinate[0], a_select, r, terminal, state2_stack_arr)

            #print('state_stack shape : ', np.shape(state_stack_arr), '  | reward : ', r, '  action : ', a_raw,
            #      ' | predicted_q : ', critic.predict(state_stack_arr, a_raw, a_descrete))
            #input()


            state_stack = state2_stack
            episode_reward += r


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

            actor_SPG = actorNetwork_SPG(sess, args['screen_size'], args['action_dim'], action_bound,
                                 args['tau'], args['minibatch_size'], args['actor_lr'], actor.get_trainable_params_num())

            # sess, screen_size, action_dim, learning_rate, tau, gamma, num_actor_vars, minibatch_size
            critic = criticNetwork(sess, args['screen_size'], actor.get_trainable_params_num(),
                                   actor_SPG.get_trainable_params_num(), args['tau'], args['critic_lr'],
                                   args['gamma'], args['action_dim'])

            replay_buffer = ReplayBuffer(buffer_size=args['buffer_size'])

            train(sess, env, actor, actor_SPG, critic, args, replay_buffer)



if __name__=="__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('--map_name', default='DefeatRoaches')
    parser.add_argument('--screen_size', default=64)
    parser.add_argument('--action_dim', default=2)
    parser.add_argument('--minimap_size', default=64)
    parser.add_argument('--step_mul', default=8)
    parser.add_argument('--max_episode_step', default=200)
    parser.add_argument('--episode', default=100000)
    parser.add_argument('--tau', default=0.01)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--buffer_size', default=100000)
    parser.add_argument('--minibatch_size', default=86)
    parser.add_argument('--load_model', default=True)
    parser.add_argument('-saved_model_directory', default='./results/save_model/900_ep')
    parser.add_argument('--summary_dir', default='./results/tensorboard')
    parser.add_argument('--train_start', default=10000)

    parser.add_argument('--actor_lr', default=0.001)
    parser.add_argument('--critic_lr', default=0.01)

    flags.FLAGS(sys.argv)

    args = vars(parser.parse_args())

    main(args)