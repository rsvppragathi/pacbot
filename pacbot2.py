import gym
import numpy as np
import tensorflow as tf
import tflearn
import sys
import random
import timesteps

game_name = "MsPacman-v0"

Training_Max = 40000
T = 0
action_repeat = 4

learning_rate = 0.001

gamma = 0.99

final_epsilon = 0.1
epsilon = 1.0

action_space = 9

sum_interval = 5000


##Justin
def build_dqn(num_actions, action_repeat):

    inputs = tf.placeholder(tf.float32,[None, action_repeat, 84, 84])

    net = transpose(inputs, [0,2,3,1])
    net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu')
    net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
    net = tflearn.fully_connected(net, 256, activation='relu')
    q_values = tflearn.fully_connected(net, num_actions)
    return inputs, q_values

##Micha
class AtariEnvironment(object):

    def __init__(self, gym_env, action_repeat):
        self.env = gym_env
        self.action_repeat = action_repeat

        self.gym_actions = range(gym_env.action_space.n)
        self.state_buffer = deque()


    def get_initial_state(self):
        self.state_buffer = deque()

        observation = self.env.reset()
        observation = self.get_preprocessed_frame(observation)
        obs_array = np.stack([observation for i in range(self.action_repeat)], axis=0)

        for i in range(self.action_repeat-1):
            self.state_buffer.append(observation)
        return obs_array

    def get_preprocessed_frame(self, observation):
        return resize(rgb2gray(observation), (110, 84))[13:110 - 13, :]

    def step(self, action_index):
        observation, reward, terminal, info = self.env.step(self.gym_actions[action_index])
        observation = self.get_preprocessed_frame(observation)

        previous_frames = np.array(self.state_buffer)
        obs_list = np.empty((self.action_repeat, 84, 84))
        obs_list[:self.action_repeat-1, :] = previous_frames
        obs_list[self.action_repeat-1] = observation

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(observation)

        return obs_list, reward, terminal, info

##ALL
def actor_learner(env, session, graph_ops, num_actions,summary_ops, saver):

    inputs = graph_ops["inputs"]
    q_values = graph_ops["q_values"]
    target_inputs = graph_ops["target_inputs"]
    target_q_values = graph_ops["target_q_values"]
    reset_target_network_params = graph_ops["reset_target_network_params"]
    x = graph_ops["x"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]

    summary_placeholders, assign_ops, summary_op = summary_ops

    env = AtariEnvironment(env,action_repeat)

    s_grad = []
    x_grad = []
    y_grad = []
    t = 0
    while T < Training_Max:
        obs_array = env.get_initial_state()
        terminal = False

        ep_reward = 0
        q_max = 0
        ep_t = 0

        while True:
            readout_t = q_values.eval(session=session, feed_dict={s: [s_t]})

            action_array = np.zeros([action_space])
            if random.random() <= epsilon:
                move = random.randrange(action_space)
            else:
                move = np.argmax(readout_t)
            a_t[move] = 1

            if epsilon > final_epsilon:
                epsilon -= (inital_epsilon - final_epsilon) / anneal_epsilon_timesteps

            observation, reward, terminal, info = env.step(move)

            readout_j1 = target_q_values.eval(session, feed_dict = {st : [s_t1]})

            clipped_reward = np.clip(reward, -1, 1)

            if terminal:
                y_grad.append(clipped_reward)
            else:
                y_grad.append(clipped_reward + gamma * np.argmax(readout_j1))

            x_grad.append(x_t)
            s_grad.append(s_t)

            s_t = s_t1
            T += 1
            t += 1

            ep_reward += reward
            q_max += np.max(readout_t)

            if T % I_target == 0:
                session.run(reset_target_network_params)
            if t % I_AsyncUpdate == 0 or terminal:
                if s_batch:
                    session.run(grad_update, feed_dict = {y: y_batch, x_batch, s_batch})

                s_grad = []
                x_grad = []
                y_grad = []

            if t % checkpoint_interval == 0:
                saver.save(session, "qlearning.ckpt", global_step = t)

            if terminal:
                stats = [ep_reward, q_max/float(ep_t), epsilon]
                for i in range(len(stats)):
                    session.run(assign_ops[i],{summary_placeholders[i]: float(stats[i])})
            break



# """
# The graph is used to build a dict with certain values.
# We build one q network and one target q network.
# q network is used to continually update with scores.
# target q network is only occasionally updated, to create a slower and more
# stabile learning process.
# """


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

    graph_ops = {"inputs" : inputs,"q_values" : q_values, "target_inputs" : target_inputs, "target_q_values" : target_q_values, "reset_target_network_params" : reset_target_network_params, "x" : x, "y" : y, "grad_update" : grad_update}
    return graph_ops


"""
This is used for visualisation, dont know if we need that.
Used to continually save information about reward, qmax, and epsilon.
"""
##Robert
def build_summaries():
    scalar_summary = tf.summary.scalar
    reward = tf.Variable(0.)
    q_max = tf.Variable(0.)
    log_epsilon = tf.Variable(0.)

    scalar_summary("Reward", reward)
    scalar_summary("Qmax Value", q_max)
    scalar_summary("Epsilon", log_epsilon)

    summary_vars = [reward, q_max, log_epsilon]
    summary_placeholders = [tf.placeholder("float")
                            for i in range(len(summary_vars))]
    assign_ops = [summary_vars[i].assign(summary_placeholders[i])
                  for i in range(len(summary_vars))]
    summary_op = merge_all_summaries()
    return summary_placeholders, assign_ops, summary_op



"""
This method is used to train our model by setting an environment for each
thread, initializing the variables, initializing the target netwrok weights,
and starting the action learner threads. Summary of the training statistics
are printed as it learns.
"""

def train_model(session, graph_ops, num_actions, saver):
    env = gym.make(game)

    summary_ops = build_summaries()
    summary_op = summary_ops[-1]

    session.run(tf.global_variables_initializer())
    session.run(graph_ops["reset_target_network_params"])

    writer = tf.summary.FileWriter(summary_dir + "/qlearning" + session.graph)

    actor_learner(env, session, graph_ops, summary_ops, saver)

    previous_time_step = 0

    while True:

        if show_training:
            for env in envs:
                env.render()
        now = time.time()
        if now - previous_time > sum_interval:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            previous_time_step = now



"""
This method is for evaluating the model. The graph_ops varible that is passed
to it is unpacked, and the reward state is printed untilthe agent is done
learning.
"""
##Pragathi
def test_model(session, graph_ops, saver):
    saver.restore(session, model_path)

    env = gym.make(game)
    monitor_env = gym.wrappers.Monitor(monitor_env, "qlearning/eval", force=True)

    inputs = graph["inputs"]

    #network
    q_values = graph["q_values"]

    env = AtariEnvironment(monitor_env, action_repeat)

    #number of evaluation runs
    for i in range(10):
        #first frame
        obs = env.get_initial_state()

        #counter for reward
        episode_reward = 0

        #we get terminal from step()
        terminal = False

        while not terminal:
            monitor_env.render()
            readout_t = q_values.eval(session, feed_dict={s : [s_t]})
            #make move with highest score
            move = np.argmax(readout_t)
            observation, reward, terminal, _ = env.step(move)
            obs = observation
            episode_reward += reward

        print(episode_reward)
    monitor_env.monitor.close()

"""
This method is udes for setting the variables used in various methods. This
includes num_actions, graph_ops, and saver.
"""

#Start a session for easier control of actions.
#Build graph. Creates dict contatining network and target network.
#Choose between training and testing.

def main(_):

    with tf.Session() as session:
        graph = build_graph(action_space)
        saver = tf.train.Saver(max_to_keep=5)

    if sys.argv[1] == 'training':
        train_model(session, graph, saver)

    elif sys.argv[1] == 'testing':
        test_model(session, graph, saver)

    else:
        print('Pick training or testing by writing: python pacbot2.py training/testing')


#Code to start the program, call main.
if __name__ == "__main__":
    tf.app.run()
