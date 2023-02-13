from environment import base_env
import numpy as np

class NegativeLawOfEffectEnv(base_env.BaseBehavioralEnv):
    def __init__(self):
        self.behavior_rate1 = 0
        self.behavior_rate2 = 0 

        self.reinforcement_rate1 = 0.1
        self.reinforcement_rate2 = 0.9

        self.n_trials1 = 0
        self.n_trials2 = 0

    def get_reward(self, action):
        print(action)
    
    def update(self, state, action):
        print(state, action)
    
    def behavioral_model(self):
        self.n_trials += 1
        if self.reinforcement_rate1 > np.random.rand():

            self.behavior_rate1 += (1/self.n_trials1)*(1 + self.behavior_rate1)
        
        else:
            self.behavior_rate2 += (1/self.n_trials2)*(1 + self.behavior_rate2)
        
        print(self.behavior_rate1, self.behavior_rate2)




    
    def restart(self):
        print('restart')
    