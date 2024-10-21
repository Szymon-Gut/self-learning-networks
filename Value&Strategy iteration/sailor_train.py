import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf
import itertools

number_of_episodes = 1000                   # number of training epizodes (multi-stage processes) 
gamma = 1.0                               # discount factor


file_name = 'map_small.txt'
#file_name = 'map_easy.txt'
#file_name = 'map_big.txt'
#file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained action-value table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

strategy = np.random.randint(low=1,high=5,size=np.shape(reward_map))  # random strategy
random_strategy_mean_reward = np.mean(sf.sailor_test(reward_map,strategy,1000))
sf.draw(reward_map,strategy,"random_strategy mean reward = " + str(random_strategy_mean_reward))


# miejsce na algorytm uczenia - modelem jest tablica Q
# (symulację epizodu można wziąć z funkcji sailor_test())
# ............................

# Iteracja wartości
def value_iteration(no_of_episodes):
    for episode in range(no_of_episodes):
        print(f'Episode: {episode+1}')
        alpha = 1/(episode+1)
        for new_state in itertools.product(range(num_of_rows), range(num_of_columns)):
            for action in range(1,5):
                terminal_step = False
                state = new_state
                epsiode = []
                num_of_steps = 0


                new_state, reward = sf.environment(state, action, reward_map)
                epsiode.append((state, action, reward))
                state = new_state
                num_of_steps += 1


                if state[1] >= (num_of_columns-1):
                    terminal_step = True

                while not terminal_step:
                    action = np.argmax(Q[state[0]][state[1]]) + 1
                    new_state, reward = sf.environment(state, action, reward_map)
                    epsiode.append((state, action, reward))
                    state = new_state
                    num_of_steps += 1


                    if (state[1] >= num_of_columns-1) or  num_of_steps >= num_of_steps_max:
                        terminal_step = True
                G = 0
                visited_states = set()

                for t in range(len(epsiode)-1, -1, -1):
                    state, action, reward = epsiode[t]
                    G = gamma*G + reward
                    if (tuple(state), action) not in visited_states:
                        visited_states.add((tuple(state), action))
                        Q[state[0]][state[1]][action-1] += alpha*(G - Q[state[0]][state[1]][action-1])
    strategy = np.argmax(Q, axis=2) + 1
    strategy[:,num_of_columns-1] = 0 
    sf.sailor_test(reward_map, strategy, 1000)
    sf.draw(reward_map,strategy,"best_strategy_for_value_iteration")

def strategy_iteration():
    strategy = np.random.randint(low=1,high=5,size=np.shape(reward_map))
    Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)
    iteration = 0
    while True:
        iteration += 1
        print('Strategy iteration no: ', iteration)
        strategy_pom = strategy.copy()
        for init_state in itertools.product(range(num_of_rows), range(num_of_columns)):
            for init_action in range(1,5):
                rewards = []
                for episode in range(number_of_episodes):
                    G = 0
                    terminal_step = False
                    num_of_steps = 0
                    new_state, reward = sf.environment(init_state, init_action, reward_map)
                    state = new_state
                    G += (gamma**num_of_steps)*reward
                    num_of_steps += 1
                    if state[1] >= (num_of_columns-1):
                        terminal_step = True

                    while not terminal_step:
                        action = strategy[state[0]][state[1]]
                        new_state, reward = sf.environment(state, action, reward_map)
                        state = new_state
                        G += (gamma**num_of_steps)*reward
                        num_of_steps += 1
                        if (state[1] >= num_of_columns-1) or  num_of_steps >= num_of_steps_max:
                            terminal_step = True
                    rewards.append(G)   
                Q[init_state[0]][init_state[1]][init_action-1] = np.mean(rewards)
        strategy = np.argmax(Q, axis=2) + 1
        strategy[:,num_of_columns-1] = 0
        if np.array_equal(strategy, strategy_pom):
            break
    sf.sailor_test(reward_map, strategy, 1000)
    sf.draw(reward_map,strategy,"best_strategy_for_strategy_iteration")


strategy_iteration()
# value_iteration(1000)
                    