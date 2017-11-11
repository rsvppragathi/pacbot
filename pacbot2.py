
##Justin
def build_dqn(num_actions, action_repeat):
    pass


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
    pass



"""
The graph is used to build a dict with certain values.
We build one q network and one target q network.
q network is used to continually update with scores.
target q network is only occasionally updated, to create a slower and more
stabile learning process.

"""
##Robert
def build_graph(num_actions):
    pass


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
    pass

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
