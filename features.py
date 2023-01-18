import numpy as np


def get_bias(state, env):
    return 1

def get_last_action(state, env):
    return env.current_action

#for matching to sample
def get_model_stimuli(state, env):
    z = np.zeros(3)
    z[0] = 1*(1 - env.is_model_present)
    z[env.model_stimuli_number] = 1*env.is_model_present
    return z

#for matching to sample
def get_if_last_step(state, env):
    print('get if last step state')
    print(state)
    if np.any(state[0, :] != 0):
        return 1
    return 0

def get_if_x_in_same_pos_as_y(state, x, y):
    row_x, col_x = np.where(state == x)
    x_coors = (row_x[0], col_x[0])
    rows_y, cols_y = np.where(state == y)
    ys_coors = np.array([rows_y, cols_y]).T
    return 1 if (x_coors == ys_coors).all(-1).any() else 0

def get_if_x_in_same_col_as_y(state, env):
    #print(state)
    rows_y, cols_y = np.where(state == env.block_number)
    
    row, col = np.where(state == env.agent_number)
    
    return 1 if col in cols_y else 0

def get_block_coors_in_same_col_as_agent(state, env):

    rows_y, cols_y = np.where(state == env.block_number)
    row, col = np.where(state == env.agent_number)
    row, col = np.where(state[:, col] == env.block_number)
    return row[0]
    
def get_distance_between_agent_and_block_based_feature(state, env):
    if get_if_x_in_same_col_as_y(state, env):
        rows_y, cols_y = np.where(state == env.block_number)
    
        row, col = np.where(state == env.agent_number)
        block_row = get_block_coors_in_same_col_as_agent(state, env)
        #print('rowwww: {}'.format(row))
        #print('ques es: {}'.format(1/(row - block_row) ))
        return 1/(row[0] - block_row) 
    return 0

def get_distance_to_nearest_hole_left(state, env):
    row, col = np.where(state == env.agent_number)
    #rows_y, cols_y = np.where(state == env.block_number)
    #print('pille row col')
    #print(row, col)
    try:
        array_with_cols_without_blocks = np.where(np.all(env.block_number != state[:, :col[0]+1], axis=0))
#         print('array_with_cols_without_blocks')
#         print(array_with_cols_without_blocks)

        if len(array_with_cols_without_blocks[0]) > 0:
            idx_of_nearest_hole = np.argmin(np.abs(array_with_cols_without_blocks[0] - col[0]))
            distance_to_hole_to_the_left = np.abs(array_with_cols_without_blocks[0][idx_of_nearest_hole] - col[0]) + 1
            return 1/distance_to_hole_to_the_left
    except:
        pass
    return 0
def get_distance_to_nearest_hole_right(state, env):
    row, col = np.where(state == env.agent_number)
    #rows_y, cols_y = np.where(state == env.block_number)
    try:
        array_with_cols_without_blocks = np.where(np.all(env.block_number != state[:, col[0]:], axis=0))
        if len(array_with_cols_without_blocks[0]) > 0:
            idx_of_nearest_hole = np.argmin(array_with_cols_without_blocks[0])
            distance_to_hole_to_the_right = array_with_cols_without_blocks[0][idx_of_nearest_hole] + 1
            return 1/distance_to_hole_to_the_right
    except:
        pass
    return 0


        
#def get_
class Learn:
    def __init__(self, method='sarsa'):
        
        self.method = method
    
    def set_agent(self, agent):
        self.agent = agent
        
    def calculate_q_values(self, state):
        feats = self.agent.get_x_from_s(state)
        #print('feats')
        #print(feats)
        q_values = np.dot(self.agent.w, feats)
#         print('state')
#         print(state)
#         print('feats')
#         print(feats)
#         print('q_values')
#         print(q_values)
        self.agent.current_qs = q_values
        return q_values
    def calculate_sarsa_q_next_value(self, q_next_values):
        sarsa_next_q_value = np.max(q_next_values) if self.agent.epsilon < np.random.rand() else q_next_values[np.random.choice(self.agent.n_actions)]
        return sarsa_next_q_value
    def calculate_expected_sarsa_q_next_value(self,q_next_values):
        total_actions_taken = np.sum(self.agent.actions_counter)
        expected_sarsa_next_q_value = np.dot(self.agent.actions_counter/total_actions_taken, q_next_values)
        return expected_sarsa_next_q_value
    def calculate_q_learning_q_next_value(self, q_next_values):
        q_learning_next_q_value = np.max(q_next_values)
        return q_learning_next_q_value
    def get_delta(self, state, next_state, r, done):
        q_values = self.calculate_q_values(state)
        q_next_values = self.calculate_q_values(next_state)
        q_target = q_values.copy()
        if self.method == 'sarsa':
            next_q_value = self.calculate_sarsa_q_next_value(q_next_values)#np.max(q_next_values) if agent.epsilon < np.random.rand() else q_next_values[np.random.choice(agent.n_actions)]
        elif self.method == 'q-learning':
            next_q_value = self.calculate_q_learning_q_next_value(q_next_values)#np.max(q_next_values) if agent.epsilon < np.random.rand() else q_next_values[np.random.choice(agent.n_actions)]
        elif self.method == 'expected-sarsa':
            next_q_value = self.calculate_expected_sarsa_q_next_value(q_next_values)
#         print('q_target')
#         print(q_target)
        q_target[self.agent.current_action] = r + (1 - done)*self.agent.gamma*next_q_value
        return q_target - q_values
