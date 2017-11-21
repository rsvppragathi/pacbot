import gym
import numpy as np
import tensorflow as tf
import tflearn

game_name = "MsPacman-v0"

training_max = 40000

action_repeat = 4

learning_rate = 0.001

gamma = 0.99

epsilon = 0.1




##Justin
def build_dqn(num_actions, action_repeat):

    inputs = tf.placeholder(tf.float32, [None, action_repeat, 84, 84])

    net = transpose(inputs, [0,2,3,1])
    net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu')
    net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
    net = tflearn.fully_connected(net, 256, activation='relu')
    q_values = tflearn.fully_connected(net, num_actions)
    return inputs, q_values

##Micha
class AtariEnvironment(object):

    def __init__(self, gym_env, action_repeat):
        pass
    def get_initial_state(self):
        pass
    def get_preprocessed_frame(self, observation):
        pass
    def step(self, action_index):
        pass

##ALL
def actor_learner(env, graph_ops, num_actions,summary_ops, saver):


"""
The graph is used to build a dict with certain values.
We build one q network and one target q network.
q network is used to continually update with scores.
target q network is only occasionally updated, to create a slower and more
stabile learning process.

"""
##Robert
def build_graph(num_actions):
    inputs, q_values = build_dqn(num_actions, action_repeat)
    network_params = tf.trainable_variables()

    target_inputs, target_q_values = build_dqn(num_actions, action_repeat)
    target_network_params = tf.trainable_variables()[len(network_params):]

    reset_target_network_params = \
        [target_network_params[i].assign(network_params[i])
         for i in range(len(target_network_params))]

    x = tf.placeholder(tf.float32, [None, num_actions])
    y = tf.placeholder(tf.float32, [None])

    action_q_values = tf.reduce_sum(tf.multiply(q_values, x), reduction_indices=1)

    cost = tflearn.mean_square(action_q_values, y)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params)

    graph_ops = {"inputs" : inputs,
                "q_values" : q_values,
                "target_inputs" = target_inputs,
                "target_q_values" = target_q_values,
                "reset_target_network_params" = reset_target_network_params,
                "x" = x,
                "y" = y,
                "grad_update" = grad_update}
    return graph_ops


"""
This is used for visualisation, dont know if we need that.
Used to continually save information about reward, qmax, and epsilon.
"""
##Robert
def build_summaries():
    pass

"""
This method is used to train our model by setting an environment for each
thread, initializing the variables, initializing the target netwrok weights,
and starting the action learner threads. Summary of the training statistics
are printed as it learns.
"""
##Pragathi
def train(session, graph_ops, num_actions, saver):
    env = gym.make(game)

    session.run(tf.global_variables_initializer())
    session.run(graph_ops["reset_target_network_params"])

    last_summary_time = 0
    while True:
        for env in envs:
            env.render()



"""
This method is for evaluating the model. The graph_ops varible that is passed
to it is unpacked, and the reward state is printed untilthe agent is done
learning.
"""
##Pragathi
def evaluation(session, graph_ops, saver):

"""
This method is udes for setting the variables used in various methods. This
includes num_actions, graph_ops, and saver.
"""
##Pragathi
def main(_):
    pass
