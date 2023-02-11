from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    
    @abstractmethod
    def __init__(self, env, learn, features, n_actions= 3, alpha=0.1, dec_alpha=1, min_alpha=0.01, epsilon=0.1, dec_epsilon=0.999, min_epsilon=0.001, gamma=0.9):
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
        
        
        self.current_qs = None
        
        self.alpha = alpha
        self.dec_alpha = dec_alpha
        self.min_alpha = min_alpha
        self.epsilon = epsilon
        self.dec_epsilon = dec_epsilon
        self.min_epsilon = min_epsilon
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

    @abstractmethod
    def update_w(self, state, action, next_state, reward, done):
        pass
    
    @abstractmethod
    def count_action(self, action):
        pass
    
    @abstractmethod
    def get_agent_action(self, state):
        pass
    
    def restart(self):
        self.current_action = None
