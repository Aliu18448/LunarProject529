import gymnasium as gym
import tensorflow as tf
import pandas as pd
from keras import layers, Model, optimizers
from collections import defaultdict
from Agent import neural_network

Q = defaultdict(float)
gamma = 0.99  # Discounting factor
alpha = 0.5  # soft update param


env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode = "human")

observation, info = env.reset()
# observation: what the agent can "see" - vechile position, velocity, etc.
# info: extra debugging information (usually not needed for basic learning)

print(f"Starting observation: {observation}")
# Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
observation_size = env.observation_space.shape[0]

def create_q_model(obs_size, act_size):
    inputs = layers.Input(shape=(obs_size,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(act_size, activation=None)(x)
    return Model(inputs=inputs, outputs=outputs)

# Initialize networks
observation_size = env.observation_space.shape[0]
act_size = 2 # Assuming 2 based on your layers_sizes [32, 32, 2]

q_primary = create_q_model(observation_size, act_size)
q_target = create_q_model(observation_size, act_size)

# Sync weights initially
q_target.set_weights(q_primary.get_weights())

optimizer = optimizers.Adam(learning_rate=0.001)

def train_step(states, actions, rewards, states_next, done_flags, gamma=0.99):
    # 1. Calculate Target Q-values (Bellman Equation)
    # We use q_target for the next states
    next_q_values = q_target(states_next)
    max_q_next = tf.reduce_max(next_q_values, axis=-1)
    
    # Target y = r + gamma * max(Q_target) * (1 - done)
    y_targets = rewards + (1.0 - done_flags) * gamma * max_q_next

    # 2. Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Get Q-values for current states
        current_q_values = q_primary(states)
        
        # Select the Q-values for the specific actions taken
        action_masks = tf.one_hot(actions, act_size)
        preds = tf.reduce_sum(current_q_values * action_masks, axis=-1)
        
        # Calculate Mean Squared Error Loss
        loss = tf.reduce_mean(tf.square(y_targets - preds))

    # 3. Apply Gradients
    variables = q_primary.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return loss

target_weights = q_target.get_weights()
primary_weights = q_primary.get_weights()

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = push cart left, 1 = push cart right
    action = env.action_space.sample()  # Random action for now - real agents will be smarter!

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: 
    # terminated:
    # truncated:

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()

