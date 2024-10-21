import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

number_of_episodes = 30000                # number of training epizodes (multi-stage processes)                            # discount factor


# file_name = 'map_simple.txt'
#file_name = 'map_easy.txt'
file_name = 'map_big.txt'
#file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
np.random.seed(0)
class QLAgent:
    def __init__(self, n_actions, env_rows, env_cols, alpha=0.01, gamma=0.99, epsilon=1, epsilon_decay=0.999,
                 min_epsilon=0.01) -> None:
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.env_rows = env_rows
        self.env_cols = env_cols
        self.q_table = np.zeros([self.env_rows, self.env_cols, self.n_actions], dtype=float)
    
    def get_action(self, state) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions) + 1
        else:
            return np.argmax(self.q_table[state[0], state[1]]) + 1
    
    def update(self, state, action, reward, new_state):
        self.q_table[state[0], state[1], action - 1] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[new_state[0], new_state[1]]) - self.q_table[state[0], state[1], action - 1]
        )
    def update_exploration(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# miejsce na algorytm uczenia - modelem jest tablica Q
# (symulację epizodu można wziąć z funkcji sailor_test())
# ............................
agent = QLAgent(n_actions=4, env_rows=num_of_rows, env_cols=num_of_columns, alpha=0.01, gamma=1, epsilon=1, epsilon_decay=0.9999, min_epsilon=0.01)
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)
for episode in range(number_of_episodes):
    i = np.random.randint(num_of_rows)
    state = [i, 0]
    sum_of_rewards[episode] = 0
    step = 0
    while step <= num_of_steps_max:
        action = agent.get_action(state)
        new_state, reward = sf.environment(state, action, reward_map)
        step += 1
        sum_of_rewards[episode] += reward
        agent.update(state, action, reward, new_state)
        state = new_state
        if (state[1] >= num_of_columns-1):
            break
    agent.update_exploration()
    if episode % 100 == 0:
        print(f"Episode {episode}: {sum_of_rewards[episode]}. Epsilon: {agent.epsilon}")
strategy = np.argmax(agent.q_table, axis=2) + 1
print(agent.q_table)
sf.sailor_test(reward_map, strategy, 2000) 
sf.draw_strategy(reward_map,strategy,f"best_strategy_{file_name}")

# Simple
# alpha=0.01, gamma=1, epsilon=1, epsilon_decay=0.9999, min_epsilon=0.01, score=7.76, 30000 epizodów
