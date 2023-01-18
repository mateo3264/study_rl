import numpy as np
from keras.models import Sequential

class ReplayBuffer:
    def __init__(self, input_size, n_actions, mem_size=1000000):
        self.mem_size = mem_size
        self.mem_counter = 0
        self.state_memory = np.zeros((mem_size, input_size))
        self.action_memory = np.zeros((mem_size, n_actions))
        self.next_state_memory = np.zeros((mem_size, input_size))
        self.reward_memory = np.zeros(mem_size)
        self.done_memory = np.zeros(mem_size)
    
    def store_transition(self, state, action, next_state, reward, done):
        index = self.mem_counter % self.mem_size 
        self.state_memory[index] = state
        self.action_memory[index] = [0, 0, 0]
        self.action_memory[index][action] = 1
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.done_memory[index] = done
        
        self.mem_counter += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        indexes = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[indexes]
        actions = self.action_memory[indexes]
        next_states = self.next_state_memory[indexes]
        rewards = self.reward_memory[indexes]
        dones = self.done_memory[indexes]
        
        return states, actions, next_states, rewards, dones


def build_dqn(n_inputs, fc1, fc2, n_actions, learning_rate=0.01):
    model = Sequential([
        Dense(fc1, input_shape=(n_inputs, )),
        Activation('relu'),
        Dense(fc2),
        Activation('relu'),
        Dense(n_actions)
        
    ])
    
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    
    return model


class Agent:
    def __init__(self, input_size, alpha=0.001, epsilon=1, gamma=0.9, n_actions=3, batch_size=32, fc1=64, fc2=64):
        self.input_size = input_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.n_actions = n_actions
        self.current_action = None
        self.actions_counter = np.array([0 for _ in range(self.n_actions)])
        
        self.q_eval = build_dqn(self.input_size, fc1, fc2, 3, alpha)
        
        self.replay_buffer = ReplayBuffer(input_size, n_actions)
        
        self.batch_size = batch_size 
    
    def count_action(self, action):
        self.actions_counter[action] += 1
        
    
    def remember(self, state, action, next_state, reward, done):
        self.replay_buffer.store_transition(state, action, next_state, reward, done)
    
    def get_agent_action(self, state):
        state = state[np.newaxis, :]
        
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            q_values = self.q_eval.predict(state)
            action = np.argmax(q_values)
        self.current_action = action
        return action
    
    def update_w(self, state, action, next_state, reward, done):
        self.remember(state, action, next_state, reward, done)
        
        if self.replay_buffer.mem_counter > self.batch_size:
            states, actions, next_states, rewards, dones = self.replay_buffer.sample_buffer(self.batch_size)
        
            action_idxs = np.where(actions == 1)[1]


            row_idxs = np.arange(self.batch_size)

            q_values = self.q_eval.predict(states)
            q_next_values = self.q_eval.predict(next_states)
            q_targets = q_values.copy()

            q_targets[row_idxs, action_idxs] = rewards + (1 - dones)*self.gamma*np.max(q_next_values, axis=1)

            _ = self.q_eval.fit(states, q_targets)
        