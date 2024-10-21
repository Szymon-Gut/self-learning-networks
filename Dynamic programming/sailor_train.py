# Skrypt do trenowania strategii żeglarza w postaci tablicy Q 

import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

number_of_episodes = 1000                   # number of training epizodes (multi-stage processes) 
gamma = 1.0                                 # discount factor

file_name = 'map_small.txt'
# file_name = 'map_simple.txt'
#file_name = 'map_easy.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt' #195.54616000000004

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained action-value table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

# strategy = np.random.randint(low=1,high=5,size=np.shape(reward_map))  # random strategy
# random_strategy_mean_reward = np.mean(sf.sailor_test(reward_map,strategy,1000))
# sf.draw_strategy(reward_map,strategy,"random_strategy_average_reward_=_" + str(np.round(random_strategy_mean_reward,2)))


# miejsce na algorytm uczenia - modelem jest tablica Q
# (symulację epizodu można wziąć z funkcji sailor_test())
# ............................
# ............................
def transition_model(state, action, reward_map):
    num_of_rows, num_of_columns = reward_map.shape
    prob_side = 0.12
    prob_back = 0.06
    prob_action = 1 - 2 * prob_side - prob_back
    wall_colid_reward = -0.04
    trajectories = []

    # zdefiniowanie zmiany położenia w zależności od akcji
    action_moves_probabilities = {
        1: [((0, 1), prob_action), ((-1, 0), prob_side), ((1, 0), prob_side), ((0, -1), prob_back)],  # prawo: [prawo, góra, dół, lewo]
        2: [((-1, 0), prob_action), ((0, 1), prob_side), ((0, -1), prob_side), ((1, 0), prob_back)],  # góra: [góra, prawo, lewo, dół]
        3: [((0, -1), prob_action), ((-1, 0), prob_side), ((1, 0),prob_side), ((0, 1), prob_back)],  # lewo: [lewo, góra, dół, prawo]
        4: [((1, 0), prob_action), ((0, 1), prob_side), ((0, -1), prob_side), ((-1, 0), prob_back)]   # dół: [dół, prawo, lewo, góra]
    }
    for move, probability in action_moves_probabilities[action]:
        new_state = state + np.array(move)
        if 0 <= new_state[0] < num_of_rows and 0 <= new_state[1] < num_of_columns:
            trajectories.append((new_state, probability, reward_map[new_state[0], new_state[1]]))
        else:
            trajectories.append((state, probability, wall_colid_reward))
    return trajectories

def value_iteration(reward_map, delta_max, gamma):
    num_of_rows, num_of_columns = reward_map.shape
    iteration = 0
    stop = False
    while not stop:
        iteration += 1
        print('Value iteration no: ', iteration)
        delta = 0
        Q_pom = Q.copy()
        for i in range(num_of_rows):
            for j in range(num_of_columns):
                for action in range(1, 5):
                    if j < num_of_columns - 1:
                        state = np.array([i, j])
                        trajectories = transition_model(state, action, reward_map)
                        Q[i, j, action - 1] = sum(
                                prob * (reward + gamma * max(Q_pom[s_next[0], s_next[1]]))
                                for s_next, prob, reward in trajectories)
                        delta = max(delta, abs(Q[i, j, action - 1] - Q_pom[i, j, action - 1]))
        if delta < delta_max:
            stop = True
    strategy = np.argmax(Q, axis=2) + 1
    sf.sailor_test(reward_map, strategy, 1000)
    sf.draw_strategy(reward_map,strategy,f"best_strategy_{file_name}")

value_iteration(reward_map, 0.0001, gamma)




