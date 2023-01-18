import numpy as np
from constants import *

class FeatAgent:
    def __init__(self, env, learn, features, n_actions= 3, alpha=0.1, epsilon=0.1, gamma=0.9):
        self.env = env
        self.learn = learn
        self.learn.set_agent(self)
        self.features = features
        self.n_features = self.count_elements_of_feats()#len(features)self.
        print('self.n_features')
        print(self.n_features)
        
        self.n_actions = n_actions
        self.current_action = None
        self.actions_counter = np.array([0 for _ in range(self.n_actions)])
        
        self.w = np.random.rand(self.n_actions, self.n_features)
        self.current_qs = None
        
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
    
    def count_elements_of_feats(self):
        s = np.zeros((self.env.rows, self.env.cols))
        feat_counter = 0
        for f in self.features:
            res = f(s, self.env)
            if isinstance(res, int):
                feat_counter +=1
            else:
                feat_counter += len(res)
        
        return feat_counter
        
    def count_action(self, action):
        self.actions_counter[action] += 1
        
    def get_x_from_s(self, s):
        #f_same_col = get_if_x_in_same_col_as_y(s, env.agent_number, env.block_number)
        #feats = np.array([f(s, self.env) for f in self.features])))
        feats = []
        for f in self.features:
            res = f(s, self.env)
            if isinstance(res, np.ndarray):
                #res = *res
                feats += list(res)
            else:
                feats.append(res)
        feats = np.array(feats)
        #print('fffeats: {}'.format(feats))
        return feats#np.array([1, f_same_col])

    def calculate_v(self, s):
        x = self.get_x_from_s(s, self.env)
        
        
        return np.dot(self.w, x)

    def update_w(self, grid, action, next_grid, r, done):
        
        x = self.get_x_from_s(grid)
        #print('x: {}'.format(x))
        #v_next = self.calculate_v(next_grid, self.env)
        #v = self.calculate_v(grid, self.env)
        delta = self.learn.get_delta(grid, next_grid, r, done)
#         print('self.w')
#         print(self.w)
#         print('self.w[agent.current_action]')
#         print(self.w[agent.current_action])
#         print('delta')
#         print(delta)
#         print('x')
#         print(x)
        self.w += ALPHA*delta[:, np.newaxis]*x
    
    def get_random_action(self):
        self.current_action = np.random.randint(self.n_actions)
        return self.current_action
    
    def get_agent_action(self, s):
        if np.random.rand() < self.epsilon:
            #print('random action')
            action = np.random.choice(self.n_actions)
            self.current_action = action
             
        else:
            self.current_action = np.argmax(self.learn.calculate_q_values(s))
        
        
            
            
        return self.current_action

