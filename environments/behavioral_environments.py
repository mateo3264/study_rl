from environments import base_env


class NegativeLawOfEffectEnv(base_env.BaseBehavioralEnv):
    def __init__(self):
        pass 

    def get_reward(self, action):
        print(action)
    
    def update(self, state, action):
        print(state, action)
    
    def behavioral_model(self):
        print('behavioral model')
    
    def restart(self):
        print('restart')
    