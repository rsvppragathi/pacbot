
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

##Pragathi
def train(session, graph_ops, num_actions, saver):
    pass

##Pragathi
def evaluation(session, graph_ops, saver):


##Pragathi
def main(_):
    pass
