import gym
import universe
import random
import numpy as np
#create environment
game = 'MsPacman-v0'
env = gym.make(game)

TMax = 500
T = 0



lr = 0.8 #learning rate
y = 0.95 #epsilon
num_episodes = 2000

rList = [] #list for rewards

num_actions = env.action_space.n # = 9

obs_size = env.observation_space.shape[0]
Q = np.zeros([229,num_actions*2+2])


for _ in range(num_episodes):
    s = env.reset()
    rAll = 0
    d = False #done?
    j = 0
    while j < 99:
        j += 1
        action = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(j+1)))
        s1,r,d,_ = env.step(action)
        Q[s,action] = Q[s,action] + lr*(r + y*np.max(Q[s1,:])-Q[s,action])
        rAll += r
        s = s1
        if d:
            break
    rList.append(rAll)
    #env.render()
    #observation, reward, done, info = env.step(env.action_space.sample())
print("Score over time: "+ str(sum(rList)/num_episodes))
print("Final Q-table values: " + Q)


##Reinforcement learning model
def build_dqn(num_actions, action_repeat):
    inputs = tf.placeholder(tf.float32,[None, action_repeat,84,84])
    net = tf.transpose(inputs,[0,2,3,1])
    net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu')
    net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
    net = tflearn.fully_connected(net, 256, activation='relu')
    q_values = tflearn.fully_connected(net, num_actions)
    return inputs, q_values

class AtariEnvironment(object):
    def __init__(self, gym_env, action_repeat):
        self.env = gym_env
        self.action_repeat = action_repeat

        self.gym_actions = range(gym_env.action_space.n)
        self.state_buffer = deque()

    def get_initial_state(self):
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack([x_t for i in range(self.action_repeat)], axis=0)

        for i in range(self.action_repeat-1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_frame(self, observation):
        return resize(rgb2gray(observation),(110,84))[13:110 -13,:]


    def step(self, action_index):
        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self.get_preprocessed_frame(x_t1)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.action_repeat,84,84))
        s_t1[:self.action_repeat-1,:] = previous_frames
        s_t1[self.action_repeat-1] = x_t1

        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)
        return s_t1, r_t, terminal, info

##end of environment Wrapper
## OpenAI, GYM, Reinforcement learning, Q-Learning

## function for getting "random epsilon", used to give threads different training.
def sample_final_epsilon():
    final_epsilons = np.array([0.1,0.01,0.5])
    probabilities = np.array([0.4,0.3,0.3])

    return np.random.choice(final_epsilons,1,p=list(probabilities))[0]

def actor_learner_thread(thread_id, env, session, graph_ops, num_actions, summary_ops, saver):
    global TMax, T

    s = graph_ops["s"]
    q_values = graph_ops["q_values"]
    st = graph_ops["st"]
    target_q_values = graph_ops["target_q_values"]
    reset_target_network_params = graph_ops["reset_target_network_params"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]

    summary_placeholders, assign_ops, summary_op = summary_ops

    env = AtariEnvironment(gym_env = env, action_repeat = action_repeat)

    s_batch = []
    a_batch = []
    y_batch = []

    final_epsilon = sample_final_epsilon
    initial_epsilon = 1.0
    epsilon = 1.0

    print("Thread " + str(thread_id) + " - Final epsilon: " + str(final_epsilon))

    time.sleep(3*thread_id)
    t = 0
    while T < TMax:
        s_t = env.get_initial_state()
        terminal = False

        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0

        while True:
            readout_t = q_values.eval(session=session, feed_dict={s:[s_t]})
            a_t = np.zeros([num_actions])
            if random.random() <= epsilon:
                action_index = random.randrange(num_actions)
            else:
                action_index = np.argmax(readout_t)
            a_t[action_index] = 1

            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / anneal_epsilon_timesteps

            s_t1, r_t, terminal, info = env.step(action_index)

            readout_j1 = target_q_values.eval(session = session, feed_dict = {st : [s_t1]})
            clipped_r_t = np.clip(r_t,-1,1)
            if terminal:
                y_batch.append(clipped_r_t)
            else:
                y_batch.append(clipped_r_t + gamme * np.max(readout_j1))

            a_batch.append(a_t)
            s_batch.append(s_t)

            s_t = s_t1
            T += 1
            t += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(readout_t)

            if T % I_target == 0:
                session.run(reset_target_network_params)

            if t % checkpoint_interval == 0:
                saver.save(session, "qlearning.ckpt", global_step=t)


def build_graph():
    pass

def build_summaries():
    pass

##Training the model
def train_model():
    ##train model
    pass

def evaluation():
    ##if we have a model ready, we can go to evaluation directly
    pass
##Making it play the game
def main():
    with tf.Session() as session:
        num_actions = get_num_actions()
        graph_ops = build_graph(num_actions)
        saver = tf.train.Saver(max_to_keep=5)

        if testing:
            evaluation(session, graph_ops, saver)
        else:
            train(session, graph_ops,num_actions, saver)

if __name__ == "__main__":
    tf.app.run()
