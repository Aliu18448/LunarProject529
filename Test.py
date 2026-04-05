import gymnasium as gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from keras import layers, Model, optimizers

# --- Hyperparameters ---
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64

memory = deque(maxlen=20000) # Replay Buffer

# --- Environment Setup ---
env = gym.make("LunarLander-v3", continuous=False, render_mode="human")
observation_size = int(env.observation_space.shape[0])
action_size = int(env.action_space.n) # This is 4 for LunarLander

# --- Model Definition ---
def create_q_model(observation_size, action_size):
    inputs = layers.Input(shape=(observation_size,))
    # shape is a tuple, so can empty one side for input
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(action_size, activation=None)(x)
    return Model(inputs=inputs, outputs=outputs)

q_primary = create_q_model(observation_size, action_size)
q_target = create_q_model(observation_size, action_size)
q_target.set_weights(q_primary.get_weights())

optimizer = optimizers.Adam(learning_rate=0.001)

# --- Training Logic ---
@tf.function
def train_step(states, actions, rewards, next_states, dones):
    next_q = q_target(next_states)
    max_next_q = tf.reduce_max(next_q, axis=1)
    target_q = rewards + (gamma * max_next_q * (1.0 - dones))
    
    with tf.GradientTape() as tape:
        current_q = q_primary(states)
        action_masks = tf.one_hot(actions, action_size)
        preds = tf.reduce_sum(current_q * action_masks, axis=1)
        loss = tf.reduce_mean(tf.square(target_q - preds))
        
    grads = tape.gradient(loss, q_primary.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_primary.trainable_variables))
    return loss

# --- Main Loop ---
for episode in range(500):
    state, info = env.reset()
    state = np.reshape(state, [1, observation_size])
    total_reward = 0
    
    for time in range(1000):
        # 1. Epsilon-Greedy Action Selection
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            q_values = q_primary(state)
            action = np.argmax(q_values[0])

        # 2. Environment Step
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.reshape(next_state, [1, observation_size])
        done = terminated or truncated
        
        # 3. Store in Replay Buffer
        memory.append((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward

        # 4. Train from Experience
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            
            # Convert minibatch to tensors
            s_batch = tf.convert_to_tensor(np.vstack([x[0] for x in minibatch]), dtype=tf.float32)
            a_batch = tf.convert_to_tensor(np.array([x[1] for x in minibatch]), dtype=tf.int32)
            r_batch = tf.convert_to_tensor(np.array([x[2] for x in minibatch]), dtype=tf.float32)
            ns_batch = tf.convert_to_tensor(np.vstack([x[3] for x in minibatch]), dtype=tf.float32)
            d_batch = tf.convert_to_tensor(np.array([x[4] for x in minibatch]), dtype=tf.float32)
            
            train_step(s_batch, a_batch, r_batch, ns_batch, d_batch)

        if done:
            # Update Target Network periodically
            q_target.set_weights(q_primary.get_weights())
            print(f"Episode: {episode}, Score: {total_reward}, Epsilon: {epsilon:.2f}")
            break
            
    # Decay exploration
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

env.close()