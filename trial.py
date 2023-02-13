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

running_avg = 0
running_avg_total_rewards_data_point = 0
env_rows = 15
env_cols = 18
#env = EnvWithStimuli(env_rows, env_cols, 17)
env = MatchingToSample(latency=5)
learn = Learn('sarsa')

#, get_if_x_in_same_col_as_y, get_distance_between_agent_and_block_based_feature, get_distance_to_nearest_hole_left, get_distance_to_nearest_hole_right 
agent = FeatAgent(env, learn, [get_bias, get_last_action, get_model_stimuli, get_if_last_step], alpha=0.00001, dec_alpha=1, min_alpha=0.00001, epsilon=0.01, dec_epsilon=1, min_epsilon=0.001, gamma=0.95, n_actions=3)
#agent = FeatAgent(env, learn, [get_bias, get_if_x_in_same_col_as_y, get_distance_between_agent_and_block_based_feature, get_distance_to_nearest_hole_left, get_distance_to_nearest_hole_right], alpha=1, dec_alpha=0.99995, min_alpha=0.00001, epsilon=1, dec_epsilon=0.99995, gamma=0.95, n_actions=3) 
#agent = Agent(env, learn, [get_bias, get_last_action, get_model_stimuli, get_if_last_step], alpha=0.1, dec_alpha=0.9995, min_alpha=0.00001, gamma=0.95, n_actions=3, epsilon=0.01, dec_epsilon=1, min_epsilon=0.001, batch_size=32)
timesteps_record = []
running_avg_timesteps_record = []


total_rewards = []
running_avg_total_rewards = []
eps = 1
record = deque(maxlen=3)

# agent.w = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 1, 0, 1],
#     [0, 0, 0, 0, 1, 0, 0, 1, 1]
#      ])
for episode in range(EPISODES):
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
        #print('agent.w')
        #print(agent.w)
        timesteps +=1
        # if np.random.random() < 0.4:
        #     env.spawn_block()
        
        if learn.method == 'expected-sarsa' and np.any(agent.actions_counter == 0):
            action = np.where(0 == agent.actions_counter)[0][0]
            
        else:
            #action = agent.get_agent_action(grid.flatten())#agent.get_random_action()#
            action = agent.get_agent_action(grid)#agent.get_random_action()#
        #if timesteps == 1:
         #   action = env.model_stimuli_number
        
#         print('env.currrent_action')
#         print(env.currrent_action)
        env.current_action = action
        agent.count_action(action)
        #print('action: {}'.format(action))
#         print('grid')
#         print(grid)
        new_grid, r, done = env.update(grid, action)#, action)
        
        # if episode % 1000 == 0:
        #     if timesteps == 1:
        #         print(50*'*')
        #     print(f'timesteps: {timesteps}')
        #     print('grid :')
        #     print(grid)
        #     print(f'action : {action}')
        #     print(f'new_grid: ')
        #     print(new_grid)
        #     print(f'reward : {r}')
            
        #     print('features')
        #     see_features = get_x_from_s(grid, agent.features, env)
        #     print(see_features)
        #     print('ws')
        #     print(agent.w)
        #     print('qs')
        #     print(np.dot(see_features, agent.w.T))
        #     print('current_qs')
        #     print(agent.current_qs)
        #     print('current agent w')
        #     print(agent.w.T)

        #     print('done')
        #     print(done)
            #time.sleep(1)


        # print('timestep {}'.format(timesteps - 1))
        # print('action')
        # print(action)
        # print('feats')
        # print(agent.get_x_from_s(grid))
        # print('r')
        # print(r)
        r = env.get_reward(action)
        #agent.update_w(grid.flatten(), action,  new_grid.flatten(), r, done)
        # print('gridddd')
        # print(grid)

        #TODO: cambiar esta condición para hacer más general el código
        # if abs(r) == 1:
        #     # print('action')
        #     # print(action)
        #     # print('r')
        #     # print(r)
        #     done = True
        # print('done')
        # print(done)
        # print('agent.update_w')
        agent.update_w(grid, action,  new_grid, r, done)
        sum_rewards_in_episode += r
        add_record(record, grid, action)
        

            

        if episode % 100 == 0:
            render_grid(grid)
            render_color_grid(grid)
            #render_v_value(agent)
#         if agent.epsilon < 0.1:
#             print('grid: ')
#             print(grid)
#             print('agent w vector: {}'.format(agent.w))
#             print('Qs', agent.current_qs)
#             print('r')
#             print(r)
            #time.sleep(0.05)
        add_record(record, grid, action)
        
        grid = new_grid
        env.last_action = agent.current_action

        env.current_timestep +=1
    #add_record(grid, None)
        
    running_avg = BETA*running_avg + (1 - BETA)*timesteps
    running_avg_timesteps_record.append(running_avg)
    timesteps_record.append(timesteps)
    #print('agent w vector: {}'.format(agent.w))
    total_rewards.append(sum_rewards_in_episode)
    running_avg_total_rewards_data_point = BETA*running_avg_total_rewards_data_point + (1 - BETA)*sum_rewards_in_episode
    running_avg_total_rewards.append(running_avg_total_rewards_data_point)
    if episode % 100 == 0:
        print(50*'*')
        print('Episode: ', episode)
        print('number of timesteps {}'.format(timesteps))
        print('Avg of timesteps {}'.format(running_avg))
        print('Epsilon {}'.format(agent.epsilon))
        print('Alpha {}'.format(agent.alpha))
        #print(f'total_rewards: {total_rewards}')
        print(f'running avg total rewards: {running_avg_total_rewards[-1]}')
        #print('Recorder: ', record)
        #print('agent w vector: {}'.format(agent.w))
    
        
        #downgrade_grid(grid, action)
        #render_grid(grid)
        #time.sleep(1)

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
#plt.show()
#plt.plot(running_avg_timesteps_record)  
#plt.show()
#plt.plot(total_rewards)
#plt.show()
#plt.plot(running_avg_total_rewards)