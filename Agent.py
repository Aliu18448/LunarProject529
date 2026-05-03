#File to work on Agent

import gymnasium as gym
import tensorflow as tf
import numpy as np
from keras import layers, models, optimizers
import pandas as pd

memory = 100000 # Replay Buffer
gamma = 0.99
epsilon = 1.000
#Empty lists to record and save reward and epsilon for runs.
ramount = []
eamount = []
samount = []

# Enviroment Setup
env = gym.make("LunarLander-v3", continuous=False)
action_size = int(env.action_space.n) # Collects the number of actions available.

# Model generation and call
class Double_Network(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Layer setup
        self.l1 = layers.Dense(64, activation='relu')
        self.l2 = layers.Dense(64, activation='relu')
        self.state = layers.Dense(1, activation=None)
        self.action = layers.Dense(action_size, activation=None)
    
    def call(self, input_size):
        x = self.l1(input_size) # Dot products
        x = self.l2(x)
        state = self.state(x)
        action = self.action(x)
        # Calculates based on the action, state, and mean of layers.
        aggreate = state + (action - tf.math.reduce_mean(action, axis=1, keepdims=True)) 
        return aggreate
    
    def advantage(self, state):
        x = self.l1(state)
        x = self.l2(x)
        adv = self.action(x)
        return adv

#Handles Memory and how the model determines improvement    
class replay():
    def __init__(self):
        self.buffer_size = memory
        self.state_mem = np.zeros((self.buffer_size, *(env.observation_space.shape)), dtype = np.float32)
        self.action_mem = np.zeros((self.buffer_size), dtype = np.int32)
        self.reward_mem = np.zeros((self.buffer_size), dtype = np.float32)
        self.next_state_mem = np.zeros((self.buffer_size, *(env.observation_space.shape)), dtype = np.float32)
        self.result_mem = np.zeros((self.buffer_size), dtype=bool)
        self.pointer = 0

    def exp(self, state, action, reward, next_state, result):
        idx = self.pointer % self.buffer_size
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.result_mem[idx] = 1 - int(result)
        self.pointer += 1

    def sample(self, batch_size = 64):
        max_mem = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace = False)
        state = self.state_mem[batch]
        action = self.action_mem[batch]
        reward = self.reward_mem[batch]
        next_state = self.next_state_mem[batch]
        result = self.result_mem[batch]
        return state, action, reward, next_state, result

#Generates a Double Dueling Deep Q Network for use.    
class Agent():
    #Main class that generates a dueling double deep Q network
    def __init__(self, gamma=0.99, replace=1000, lr=0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.replace = replace
        self.action_size = action_size

        #Kept same as largest layer sizes
        self.batch_size = 64

        #Set by epsilon being a default of 1.0
        self.min_epsilon = (epsilon/100.0)

        #Decay is set to multipicative decay value
        self.epsilon_decay = 0.995
        self.trainstep = 0
        self.memory = replay()
        #Use default Adam optimizer for weights
        optim = optimizers.Adam(learning_rate=lr)

        #Use Huber Loss mixing MSE and MAE to handle losses extreme and small.
        loss = tf.keras.losses.Huber()

        #Our network to train
        self.Q_net = Double_Network()
        self.Q_net.compile(loss=loss, optimizer=optim)

        #Our network to compare with
        self.train_net = Double_Network()
        self.train_net.compile(loss=loss, optimizer=optim)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        else:
            actions = self.Q_net.advantage(np.array([state]))
            action = np.argmax(actions)
            return action
    
    def update_mem(self, state, action, reward, next_state, result):
        self.memory.exp(state, action, reward, next_state, result)

    def update_target(self):
        self.train_net.set_weights(self.Q_net.get_weights())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon) 
    
    def train(self):
        if self.memory.pointer < self.batch_size:
            return
        if self.trainstep % self.replace == 0:
            self.update_target()

        state, action, reward, next_state, result = self.memory.sample(self.batch_size)
        train = self.Q_net(state, verbose=0)
        next_state_val = self.train_net.predict(next_state, verbose=0)
        max_action = np.argmax(self.Q_net.predict(next_state, verbose=0), axis=1)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        Q_train = np.copy(train)

        #Reward value is reduced to allow for no overstrain on weight calculations.
        Q_train[batch_index, action] = (reward/10) + self.gamma * next_state_val[batch_index, max_action]*result 
        self.Q_net.train_on_batch(state, Q_train)

        self.update_epsilon()
            
        self.trainstep += 1
    
    def save_model(self):
        #Saves Models to current directory
        self.Q_net.save("DQmodel.h5")
        self.train_net.save("train_DQmodel.h5")

    def save_weights(self):
        self.Q_net.save_weights("DQmodel_weights.h5")
        self.train_net.save_weights("train_DQmodel_weights.h5")
    
    def load_model(self):
        #Loads current models.
        self.Q_net= models.load_model("DQmodel.h5", custom_objects={"Double_Network": Double_Network})
        self.train_net = models.load_model("train_DQmodel.h5", custom_objects={"Double_Network": Double_Network})

    def load_weights(self):
        self.Q_net.load_weights("DQmodel_weights.h5")
        self.train_net.load_weights("train_DQmodel_weights.h5")

if __name__ == "__main__":
    Astro = Agent()
    episodes = 1000
    for e in range(episodes):
        result = False
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        while not result:
            step_count += 1
            action = Astro.act(state)
            # Check if program is stuck in a long running instance
            if step_count % 200 == 0:
                print(f"Episode {e} is still running: Step {step_count}")
            next_state, reward, terminated, truncated, _ = env.step(action)
            result = terminated or truncated
            Astro.update_mem(state, action, reward, next_state, result)
            Astro.train()
            state = next_state
            total_reward += reward

            if result:
                print("total reward after {} episode is {} and epsilon is {}, took {} steps".format(e, total_reward, Astro.epsilon, step_count))
                ramount.append(total_reward)
                eamount.append(Astro.epsilon)
                samount.append(step_count)

    data = {
        "Total Reward" : ramount,
        "Epsilion" : eamount,
        "Step Counts": samount
    }
    df = pd.DataFrame(data)
    df.to_csv('data.txt', header=None,index=None, sep=' ', mode='a')
    Astro.save_model() 
    Astro.save_weights()
