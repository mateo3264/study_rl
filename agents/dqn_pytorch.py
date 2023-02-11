import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()#?
        self.input_dims = input_dims 
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)#?
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)#?
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)#?
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions 

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.eps_min = eps_end
        self.eps_dec = eps_dec 
        self.lr = lr 
        self.action_space = [a for a in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size 
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100


        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   cv1_dims=256,
                                   fc2_dims=256
                                   )
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
    
    def store_transition(self, state, action, next_state, reward, terminal):
        index = self.mem_cntr % self.mem_size 
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward 
        self.terminal[index] = terminal 

        self.mem_cntr += 1
    
    def choose_action(self, observation):#? cómo se entrega la observación?
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)#?Pq se envuelve la observation en una lista?
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        
        else:
            action = np.random.choice(self.action_space)
        
        return action 
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return 
        
        self.Q_eval.optimizer.zero_grad()#?

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)

        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)

        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec\
                        if self.epsilon > self.eps_min else self.eps_min

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, input_size, fc1, fc2, n_actions, lr):
        self.input_size = input_size 
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.out = nn.Linear(fc2, self.n_actions)

        self.loss = nn.MSE()
        self.optim = optim.Adam(self.parameters(), lr=lr)








import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1, fc2, n_actions):
        super(DeepQNetwork, self).__init__()

        self.lr = lr 
        self.input_dims = input_dims
        self.fc1 = fc1
        self.fc2 = fc2
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1)
        self.fc2 = nn.Linear(self.fc1, self.fc2)
        self.out = nn.Linear(self.fc2, n_actions)


        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_values = self.out(x)

        return action_values


class Agent:
    def __init__(self,input_dims, n_actions, lr=0.001, gamma=0.95, epsilon=1, dec_epsilon=0.9999, min_epsilon=0.01,
                 max_memory=1000000, batch_size=16, fc1=16, fc2=16):
        self.input_dims = input_dims
        
        self.n_actions = n_actions
        self.action_space = [a for a in range(self.n_actions)]

        self.fc1 = fc1
        self.fc2 = fc2

        self.lr = lr 
        self.gamma = gamma 
        self.epsilon = epsilon
        self.dec_epsilon = dec_epsilon
        self.min_epsilon = min_epsilon
        
        self.max_memory = max_memory
        self.mem_counter = 0
        self.batch_size = batch_size

        self.dqn = DeepQNetwork(self.input_dims, fc1, fc2, n_actions)

        self.state_memory = np.zeros((self.max_memory, *self.input_dims),
                                      dtype=np.floa32)
        self.next_state_memory = np.zeros((self.max_memory, *self.input_dims),
                                           dtype=np.floa32)
        self.action_memory = np.zeros(self.max_memory,
                                      dtype=np.int32)
        self.reward_memory = np.zeros(self.max_memory,
                                      dtype=np.floa32)
        self.terminal_memory = np.zeros(self.max_memory,
                                        dtype=np.bool)
    
    def store_transition(self, state, action, next_state, reward, terminal):
        index = self.mem_counter % self.max_memory 
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward 
        self.terminal_memory[index] = terminal

        self.mem_counter += 1
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = T.tensor([state]).to(self.dqn.device)
            action = T.argmax(self.dqn.forward(state)).item()
        
        return action

    def learn(self):
        if self.mem_counter < self.batch_size:
            return 
        
        self.dqn.optimizer.zero_grad()

        superior_lim = min(self.mem_counter, self.max_memory)
        batch_idxs = np.random.choice(superior_lim, self.batch_size, replace=False)
        ordered_idxs = np.arange(len(batch_idxs))
        states = T.tensor(self.state_memory[batch_idxs]).to(self.dqn.device)
        actions = self.action_memory[batch_idxs]
        next_states = T.tensor(self.next_state_memory[batch_idxs]).to(self.dqn.device)

        rewards = T.tensor(self.reward_memory[batch_idxs]).to(self.dqn_device)
        terminals = T.tensor(self.terminal_memory[batch_idxs]).to(self.dqn_device)

        q_vals = self.dqn.forward(states)[ordered_idxs, actions]
        q_next_vals = self.dqn.forward(next_states)
        q_next_vals[terminals] = 0.0

        q_targets = rewards + self.gamma*T.max(q_next_vals)

        loss = self.dqn.loss(q_targets, q_vals).to(self.dqn.device)

        loss.backward()#?

        self.dqn.optimizer.step()#?

        self.epsilon = self.epsilon*self.dec_epsilon if self.epsilon > self.min_epsilon \
                        else self.min_epsilon



import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1, fc2):
        super(DeepQNetwork, self).__init__()

        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1 
        self.fc2_dims = fc2 

        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.out = nn.Linear(self.fc2_dims, n_actions)

        self.loss = nn.MSELoss()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_values = self.out(x)

        return action_values


class Agent:
    def __init__(self, input_dims, n_actions, lr=0.01, gamma=0.95, epsilon=1, dec_epsilon=0.999, min_epsilon=0.01,
                 max_memory=1000000, batch_size=16, fc1=16, fc2=16):

        self.input_dims = input_dims
        
        self.n_actions = n_actions
        self.action_space = [a for a in range(n_actions)]

        self.lr = lr 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.dec_epsilon = dec_epsilon
        self.min_epsilon = min_epsilon

        self.max_memory = max_memory 
        self.mem_counter = 0 
        self.batch_size = batch_size 

        self.dqn = DeepQNetwork(self.input_dims, self.n_actions, fc1, fc2)

        self.state_memory = np.zeros((self.max_memory, *input_dims), 
                                      dtype=np.float32)
        
        self.action_memory = np.zeros((self.max_memory), 
                                      dtype=np.int8)
        
        self.next_state_memory = np.zeros((self.max_memory, *input_dims), 
                                      dtype=np.float32)
        
        self.reward_memory = np.zeros((self.max_memory), 
                                      dtype=np.int16)
        self.terminal_memory = np.zeros((self.max_memory), 
                                      dtype=np.bool)

    def store_transition(self, state, action, next_state, reward, done):
        index = self.mem_counter % self.max_memory

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = T.tensor(state).to(self.dqn.device)
            action = T.argmax(self.dqn.forward(state)).item()
        
        return action

    def learn(self):
        if self.mem_counter < self.batch_size:
            return 
        
        self.dqn.optimizer.zero_grad()
        
        max_limit = min(self.mem_counter, self.max_memory)
        
        batch_idxs = np.random.choice(max_limit, self.batch_size, replace=False)
        ordered_batch_idxs = np.arange(len(batch_idxs))

        states = T.tensor(self.state_memory[batch_idxs]).to(self.dqn.device)
        actions = self.state_memory[batch_idxs]
        next_states = T.tensor(self.next_state_memory[batch_idxs]).to(self.dqn.device)
        rewards = T.tensor(self.reward_memory[batch_idxs]).to(self.dqn.device)
        terminals = T.tensor(self.terminals_memory[batch_idxs]).to(self.dqn.device)

        q_values = self.dqn.forward(states)[ordered_batch_idxs, actions]#?
        next_q_values = self.dqn.forward(next_states)
        next_q_values[terminals] = 0.0

        q_target = rewards + self.gamma*T.max(next_q_values, dim=1)[0]

        loss = self.dqn.loss(q_target, q_values).to(self.dqn.device)

        loss.backward()
        self.dqn.optimizer.step()

        self.epsilon = self.epsilon*self.dec_epsilon if self.epsilon > self.min_epsilon \
                        else self.min_epsilon
















