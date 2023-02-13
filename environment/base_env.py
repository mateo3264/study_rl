from abc import ABC, abstractmethod 

class BaseEnv(ABC):
    @abstractmethod
    def update(self, state, action):
        pass 

    @abstractmethod
    def get_reward(self, action):
        pass 

    @abstractmethod
    def restart(self):
        pass 

class BaseBehavioralEnv(BaseEnv):
    @abstractmethod
    def behavioral_model(self):
        pass