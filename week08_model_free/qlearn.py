import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from collections import defaultdict
import os
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium.core import ObservationWrapper

class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        self._qvalues[state][action] = value

    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.0
        return max([self.get_qvalue(state, a) for a in possible_actions])

    def update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha
        q_new = (1 - learning_rate) * self.get_qvalue(state, action) + \
                learning_rate * (reward + gamma * self.get_value(next_state))
        self.set_qvalue(state, action, q_new)

    def get_best_action(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        q_vals = [self.get_qvalue(state, a) for a in possible_actions]
        return possible_actions[np.argmax(q_vals)]

    def get_action(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        if np.random.rand() < self.epsilon:
            return np.random.choice(possible_actions)
        return self.get_best_action(state)

class Discretizer(ObservationWrapper):
    def observation(self, state):
        state = np.round(state, 1)
        return tuple(state)

def make_cartpole_env():
    return gym.make('CartPole-v1', render_mode='rgb_array')

def play_and_train(env, agent, t_max=500):
    total_reward = 0.0
    s, _ = env.reset()
    for _ in range(t_max):
        a = agent.get_action(s)
        next_s, r, done, _, _ = env.step(a)
        agent.update(s, a, r, next_s)
        s = next_s
        total_reward += r
        if done:
            break
    return total_reward

def moving_average(x, span=100):
    return pd.Series(x).ewm(span=span).mean().values

# ============ CartPole Training ============
cart_env = make_cartpole_env()
cart_env = Discretizer(cart_env)
n_cart_actions = cart_env.action_space.n

cart_agent = QLearningAgent(
    alpha=0.1, epsilon=0.5, discount=0.99,
    get_legal_actions=lambda s: range(n_cart_actions))

rewards = []
for episode in range(5000):
    reward = play_and_train(cart_env, cart_agent)
    rewards.append(reward)
    cart_agent.epsilon *= 0.99995

    if episode % 500 == 0:
        print(f"[CartPole] Episode {episode}, Reward: {reward}, Epsilon: {cart_agent.epsilon:.3f}")

plt.plot(rewards, label='CartPole Rewards')
plt.plot(moving_average(rewards), label='Moving Average (100)')
plt.legend()
plt.grid()
plt.show()

# --- CartPole video ---
video_folder = "./cartpole_video"
os.makedirs(video_folder, exist_ok=True)

eval_cart_env = make_cartpole_env()
eval_cart_env = Discretizer(eval_cart_env)
eval_cart_env = RecordVideo(eval_cart_env, video_folder=video_folder, episode_trigger=lambda x: True)

state, _ = eval_cart_env.reset()
total_reward = 0.0
max_steps = 1000

for t in range(max_steps):
    action = cart_agent.get_best_action(state)
    state, reward, done, _, _ = eval_cart_env.step(action)
    total_reward += reward
    if done:
        break

eval_cart_env.close()
print(f"CartPole total reward after training: {total_reward}")
print(f"CartPole видео сохранено в папке: {video_folder}")

# ============ Taxi Training ============
taxi_env = gym.make("Taxi-v3", render_mode='rgb_array')
n_taxi_actions = taxi_env.action_space.n

taxi_agent = QLearningAgent(
    alpha=0.5, epsilon=0.25, discount=0.99,
    get_legal_actions=lambda s: range(n_taxi_actions))

def play_and_train_taxi(env, agent, t_max=500):
    total_reward = 0.0
    s, _ = env.reset()
    for _ in range(t_max):
        a = agent.get_action(s)
        next_s, r, done, _, _ = env.step(a)
        agent.update(s, a, r, next_s)
        s = next_s
        total_reward += r
        if done:
            break
    return total_reward

taxi_rewards = []
for i in range(1000):
    taxi_rewards.append(play_and_train_taxi(taxi_env, taxi_agent))
    taxi_agent.epsilon *= 0.99  # плавное уменьшение epsilon

plt.plot(taxi_rewards, label='Taxi Rewards')
plt.plot(moving_average(taxi_rewards), label='Moving Average (100)')
plt.legend()
plt.grid()
plt.show()

# --- Taxi video ---
taxi_video_folder = "./taxi_video"
os.makedirs(taxi_video_folder, exist_ok=True)

eval_taxi_env = gym.make("Taxi-v3", render_mode='rgb_array')
eval_taxi_env = RecordVideo(eval_taxi_env, video_folder=taxi_video_folder, episode_trigger=lambda x: True)

state, _ = eval_taxi_env.reset()
total_reward = 0.0
max_steps = 1000  # длинное видео

for t in range(max_steps):
    action = taxi_agent.get_best_action(state)
    state, reward, done, _, _ = eval_taxi_env.step(action)
    total_reward += reward
    if done:
        break

eval_taxi_env.close()
print(f"Taxi total reward after training: {total_reward}")
print(f"Taxi видео сохранено в папке: {taxi_video_folder}")
