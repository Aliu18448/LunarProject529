#File to showcase model

import gymnasium as gym
import numpy as np
from Agent import Agent

env = gym.make("LunarLander-v3", continuous=False, render_mode="human")
observation_size = int(env.observation_space.shape[0])
action_size = int(env.action_space.n)

test = Agent()
#Fill with dummy values for test
test.Q_net(np.zeros((1,8)))
test.train_net(np.zeros((1,8)))

test.load_weights()
test.epsilon = 0.0

episodes = 1000
for e in range(episodes):
    result = False
    state, _ = env.reset()
    total_reward = 0
    step_count = 0
    while not result:
        step_count += 1
        action = test.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        result = terminated or truncated
        state = next_state
        total_reward += reward

        if result:
            print("total reward after {} episode is {} and epsilon is {}, took {} steps".format(e, total_reward, test.epsilon, step_count))