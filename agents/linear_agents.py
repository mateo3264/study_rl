import numpy as np
from constants import *
from features import get_x_from_s
#import base_agents
from agents import base_agents 


class REINFORCEAgent(base_agents.BaseAgent):
    def __init__(self, env, learn, features, n_actions= 3, alpha=0.1, dec_alpha=1, min_alpha=0.01, epsilon=0.1, dec_epsilon=0.9999, min_epsilon=0.001, gamma=0.9):
        super(FeatAgent, self).__init__(env, learn, features, n_actions, alpha, dec_alpha, min_alpha, epsilon, dec_epsilon, min_epsilon, gamma)
        
        #se multiplica el número de las features por el n_actions para implementar stack features 
        self.n_features *= self.n_actions 
        self.theta = np.random.rand(self.n_features)
        self.rewards = []
    
    def update_w(self, state, action, next_state, reward, done):
        self.rewards.append(reward)
        x = get_x_from_s(state, self.features, self.env)
        x_sa = np.dot(x, self.theta[len(x)*action:len(x)*action + len(x)])
        sum_all_xs = np.dot(np.tile(x, self.n_actions), self.theta)

        self.theta += self.alpha*(x_sa - sum_all_xs)
    
    def get_agent_action(self, state):
        pass


class FeatAgent(base_agents.BaseAgent):
    def __init__(self, env, learn, features, n_actions= 3, alpha=0.1, dec_alpha=1, min_alpha=0.01, epsilon=0.1, dec_epsilon=0.9999, min_epsilon=0.001, gamma=0.9):
        super(FeatAgent, self).__init__(env, learn, features, n_actions, alpha, dec_alpha, min_alpha, epsilon, dec_epsilon, min_epsilon, gamma)
        self.w = np.random.rand(self.n_actions, self.n_features)/(self.n_actions*self.n_features)#np.zeros((self.n_actions, self.n_features))#

        
    def count_action(self, action):
        self.actions_counter[action] += 1

    def calculate_v(self, s):
        x = get_x_from_s(s, self.features, self.env)
        
        
        return np.dot(self.w, x)

    def update_w(self, state, action, next_state, reward, done):
        #print('update_w state before')
        #print(state)
        x = get_x_from_s(state, self.features, self.env)

        #print('update_w state after')
        #print(state)
        #print('x: {}'.format(x))
        #v_next = self.calculate_v(next_grid, self.env)
        #v = self.calculate_v(grid, self.env)
        #print('ESTOY ANTES O DESPUÉS DEL PRINT DEL ACTION 1')
        delta = self.learn.get_delta(state, next_state, reward, done)
        #print('ESTOY ANTES O DESPUÉS DEL PRINT DEL ACTION 2')
#         print('self.w')
#         print(self.w)
#         print('self.w[agent.current_action]')
#         print(self.w[agent.current_action])
#         print('delta')
#         print(delta)
        # print('from linear_agents x:')
        # print(x)
        # print('from linear_agents action:')
        # print(action)
        # print('from linear_agents  before w:')
        # print(self.w)
        # print('ALPHA*delta[:, np.newaxis]*x')
        # print(ALPHA*delta[:, np.newaxis]*x)
        # print('delta')
        # print(delta)
        
        self.w += ALPHA*delta[:, np.newaxis]*x
        # print('from linear_agents  after w:')
        # print(self.w)
        #print('EEEEEEEEEEEEEEEEEENTROOOOOOOOOOOO')

        self.epsilon = self.epsilon*self.dec_epsilon if self.epsilon > self.min_epsilon \
                        else self.min_epsilon
        self.alpha = self.alpha*self.dec_alpha if self.alpha > self.min_alpha \
                        else self.min_alpha
    
    def get_random_action(self):
        self.current_action = np.random.randint(self.n_actions)
        return self.current_action
    
    def get_agent_action(self, state):
        # print(30*'~')
        # print('from agent')
        q_values = self.learn.calculate_q_values(state)

        if np.random.rand() < self.epsilon:
            #print('random action')
            action = np.random.choice(self.n_actions)
            self.current_action = action
            #self.env.current_action = self.current_action
             
        else:
            #print('greedy action')
            
            self.current_action = np.argmax(q_values)
            #self.env.current_action = self.current_action
        
        self.current_qs = q_values
        
        
            
            
        return self.current_action

