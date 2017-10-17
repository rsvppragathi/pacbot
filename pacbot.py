import gym
import universe

env = gym.make('MsPacman-v0')
#env.configure(remotes=1)

observation_n = env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
