from environment.environments import EnvWithStimuli, MatchingToSample
from agents.linear_agents import FeatAgent
from agents.dqn import Agent
from features import *
from constants import *
from render_functions import *
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import *
from collections import deque
import time
import pandas as pd


#env = EnvWithStimuli(env_rows, env_cols, 17)



#TODO: Falta hacer alphas, epsilons etc para cada agent
def run_experiment(env, n_agents, agent_features, learning_methods, sample_size=10, n_episodes=10000, verbose=True):
    running_avg = 0
    running_avg_total_rewards_data_point = 0
    env_rows = 15
    env_cols = 18
    samples = {}
    for i in range(n_agents):


        # agent.w = np.array([
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 1, 0, 0, 1, 0, 1],
        #     [0, 0, 0, 0, 1, 0, 0, 1, 1]
        #      ])
        sample = []
        for sample_idx in range(sample_size):
            print('new element in sample')
            #agent = FeatAgent(env, learn, [get_bias, get_if_x_in_same_col_as_y, get_distance_between_agent_and_block_based_feature, get_distance_to_nearest_hole_left, get_distance_to_nearest_hole_right], alpha=1, dec_alpha=0.99995, min_alpha=0.00001, epsilon=1, dec_epsilon=0.99995, gamma=0.95, n_actions=3) 
            #agent = Agent(env, learn, [get_bias, get_last_action, get_model_stimuli, get_if_last_step], alpha=0.1, dec_alpha=0.9995, min_alpha=0.00001, gamma=0.95, n_actions=3, epsilon=0.01, dec_epsilon=1, min_epsilon=0.001, batch_size=32)
            timesteps_record = []
            running_avg_timesteps_record = []


            total_rewards = []
            running_avg_total_rewards = []
            eps = 1
            record = deque(maxlen=3)
            learn = Learn(learning_methods[i])
            features = agent_features[i]
            terminal_rewards = {'r+':0, 'r-':0}
            if learning_methods[i] == 'dqn':
                
                agent = Agent(env, learn, features, alpha=0.1, dec_alpha=0.9995, min_alpha=0.00001, gamma=0.95, n_actions=3, epsilon=0.01, dec_epsilon=1, min_epsilon=0.001, batch_size=32)
            else:
                
                
                #, get_if_x_in_same_col_as_y, get_distance_between_agent_and_block_based_feature, get_distance_to_nearest_hole_left, get_distance_to_nearest_hole_right 
                agent = FeatAgent(env, learn, features, alpha=0.1, dec_alpha=0.99994, min_alpha=0.00001, epsilon=1, dec_epsilon=0.99994, min_epsilon=0.001, gamma=0.95, n_actions=3)
            
            for episode in range(n_episodes):
                sum_rewards_in_episode = 0
                #grid = init_grid(9, 8)
                env.restart()
                agent.restart()
                #agent_coors = spawn_agent(grid)
                timesteps = 0
                done = False
                grid = env.get_grid()
            #     print('grid agent?')
            #     print(grid)
                #eps = .9*eps if eps > 0.01 else 0.01
                #agent.epsilon = .999*agent.epsilon if agent.epsilon > 0.001 else 0.01
                #agent.alpha = .9995*agent.alpha if agent.alpha > 0.00001 else 0.00001
                
                while not done:

                    timesteps +=1

                    
                    if learn.method == 'expected-sarsa' and np.any(agent.actions_counter == 0):
                        action = np.where(0 == agent.actions_counter)[0][0]
                        
                    else:
                        #action = agent.get_agent_action(grid.flatten())#agent.get_random_action()#
                        # if learning_methods[i] == 'q-learning':
                        #     action = agent.get_random_action()
                        # else:
                        action = agent.get_agent_action(grid)#agent.get_random_action()#

                    
                    env.current_action = action
                    agent.count_action(action)

                    new_grid, r, done = env.update(grid, action)
                    
                    

                    if verbose:
                        if episode % 1000 == 0:
                            print(30*'*')
                            print('sample: ')
                            print(sample_idx)
                            print('agent: ')
                            print(learning_methods[i]+str(i))
                            print(grid)
                            print(get_x_from_s(grid, agent.features, env))
                            print('action: ', action)
                            print(new_grid)
                            print(get_x_from_s(new_grid, agent.features, env))
                            print('reward: ', r)
                    if done:
                        print('r')
                        print(r)
                        if r>0:
                            terminal_rewards['r+'] +=1
                        else:
                            terminal_rewards['r-'] +=1
                        print(terminal_rewards)

                    agent.update_w(grid, action,  new_grid, r, done)
                    sum_rewards_in_episode += r
                    add_record(record, grid, action)
                    

                        

                    if episode % 100 == 0:
                        render_grid(grid)
                        render_color_grid(grid)

                    add_record(record, grid, action)
                    
                    grid = new_grid
                    env.last_action = agent.current_action

                    env.current_timestep +=1
                
                    
                running_avg = BETA*running_avg + (1 - BETA)*timesteps
                running_avg_timesteps_record.append(running_avg)
                timesteps_record.append(timesteps)
                
                total_rewards.append(sum_rewards_in_episode)
                running_avg_total_rewards_data_point = BETA*running_avg_total_rewards_data_point + (1 - BETA)*sum_rewards_in_episode
                running_avg_total_rewards.append(running_avg_total_rewards_data_point)
                if episode % 1000 == 0:
                    print(50*'*')
                    print('Episode: ', episode)
                    print('number of timesteps {}'.format(timesteps))
                    print('Avg of timesteps {}'.format(running_avg))
                    print('Epsilon {}'.format(agent.epsilon))
                    print('Alpha {}'.format(agent.alpha))
                    print(f'running avg total rewards: {running_avg_total_rewards[-1]}')

            sample.append(sum(total_rewards))
        
        samples[learning_methods[i]+str(i)] = sample

    np.save('agent_params.npy', agent.w)
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    sns.lineplot(timesteps_record, ax=ax[0, 0])
    ax[0, 0].set_title('Timesteps record')
    sns.lineplot(running_avg_timesteps_record, ax=ax[0, 1])
    ax[0, 1].set_title('Running avg timesteps record')
    sns.lineplot(total_rewards, ax=ax[1, 0])
    ax[1, 0].set_title('Total rewards')
    sns.lineplot(running_avg_total_rewards, ax=ax[1, 1])
    ax[1, 1].set_title('Running avg total rewards')
    plt.show()

    samples_pd = pd.DataFrame(samples)
    plt.figure(figsize=(10, 10))
    sns.boxplot(data=samples_pd)

    plt.show()

    return samples

