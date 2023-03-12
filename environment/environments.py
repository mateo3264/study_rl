import numpy as np
from environment import base_env


class MinEnv(base_env.BaseEnv):
    def __init__(self, rows, cols, max_n_blocks=1):
        self.rows = rows
        self.cols = cols
        
        
        self.bottom = self.rows - 1
        self.ncols = self.cols - 1
        
        self.blocks_pos = []
        self.max_n_blocks = max_n_blocks 
        
        self.agent_pos = (self.bottom, self.ncols//2)
        self.agent_number = 0.5
        
        self.block_idx = 0
        
        #self.spawn_block()
        self.block_number = 1

        self.current_timestep = 0
        
        
        
        
        
    def spawn_block(self):
        while len(self.blocks_pos) < self.max_n_blocks:
            x_block_coor = np.random.randint(self.cols)



            self.blocks_pos.append([0, x_block_coor])

            self.block_idx +=1
        
    def get_reward(self):
        #todo: cambiar a feature function
        if (np.array(self.agent_pos) == self.blocks_pos).all(-1).any():
            return -1
        return +0.1
    def clear_grid(self, y_agent, x_agent):
        grid = np.zeros((self.rows, self.cols))
        grid[y_agent, x_agent] = self.agent_number
        return grid
    
    
    def update(self, state, agent_action):
        #agent_action = agent.current_action
        new_state = state.copy()
        
        y_agent, x_agent = self.agent_pos#np.where(new_state==self.agent_number)
        new_state[y_agent, x_agent] = 0

        if agent_action == 0:

            if x_agent > 0:
                x_agent -= 1

        elif agent_action == 2:

            if x_agent < self.ncols:
                x_agent += 1
#         print('state')
#         print(state)
#         print('y_agent, x_agent')
#         print(y_agent, x_agent)
        new_state[y_agent, x_agent] = self.agent_number
        self.agent_pos = (y_agent, x_agent)


        x_coors_of_blocks_to_delete = np.where(new_state[self.bottom, :] == self.block_number)
        new_state[self.bottom, x_coors_of_blocks_to_delete] = 0
        y_block_coors, x_block_coors = np.where(new_state == self.block_number)

        new_state[y_block_coors, x_block_coors] = 0
        
        new_state[y_block_coors + 1, x_block_coors] = self.block_number
        

        if len(np.where(new_state == self.block_number)[0]) == 0:
            self.blocks_pos = []
            self.spawn_block()
            

            
            new_state = self.clear_grid(y_agent, x_agent)

            rows, cols = np.array(self.blocks_pos)[:, 0], np.array(self.blocks_pos)[:, 1]

            new_state[rows, cols] = self.block_number


        rows, cols = np.where(new_state == self.block_number)
        self.blocks_pos = np.stack((rows, cols)).T.tolist()

        reward = self.get_reward(agent_action)

        self.current_timestep
        
        return new_state, reward
    
    def get_grid(self):
        grid = np.zeros((self.rows, self.cols))

        grid[self.agent_pos[0], self.agent_pos[1]] = self.agent_number
        

        blocks_pos = np.array(self.blocks_pos)

        grid[blocks_pos[:, 0], blocks_pos[:, 1]] = self.block_number
        
        return grid
    

    
    def restart(self):
        self.agent_pos = (self.bottom, self.ncols//2)
        self.blocks_pos = []
        self.block_idx = 0
        
        self.spawn_block()

        
class MatchingToSample(base_env.BaseEnv):
    def __init__(self, latency=5, model_steps=2, comparative_steps=2, n_stimuli=2):
        self.rows = 2
        self.cols = n_stimuli
        self.latency = latency
        self.model_steps = model_steps
        self.comparative_steps = comparative_steps
        # print('self.latency')
        # print(self.latency)
        self.length_of_experiment = self.model_steps + self.latency + self.comparative_steps
        self.current_timestep = 0
        
        #una acción que no existe. Para evitar error al multiplicar w por feats
        self.last_action = -1
        self.current_action = None
        self.n_stimuli = n_stimuli
        print('self.n_stimuli')
        print(self.n_stimuli)

        
        self.model_stimuli_poss = self.spawn_model_stimuli()
        self.comparative_stimuli_poss = []
        
        
        self.comparative_stimuli_numbers = [s for s in range(1, self.n_stimuli + 1)]
        self.model_stimuli_number = np.random.choice(self.comparative_stimuli_numbers)
        self.is_model_present = True
        np.random.shuffle(self.comparative_stimuli_numbers)
        self.comparative_stimuli_poss = None#spawn_comparative_stimuli()
        self.items_in_grid = {
                                "model_stimuli":[self.model_stimuli_poss, self.model_stimuli_number],
                                "comparative_stimuli":[self.comparative_stimuli_poss, self.comparative_stimuli_numbers]
                            }

        self.tact_reward_discount = 1
        self.autoechoic_reward_discount = 1

        
        
    def spawn_model_stimuli(self):
        model_stimuli_pos = (self.rows - 1, self.cols//2)
        
        return model_stimuli_pos
    
    def spawn_comparative_stimuli(self):
        comparative_stimuli_poss = []
        for i, s in enumerate(self.comparative_stimuli_numbers):
            comparative_stimuli_poss.append((0, i))
        return comparative_stimuli_poss
    
    def get_reward(self, action):
        #print(f'dentro de get_reward self.current_action: {self.current_action}')
        # print(f'dentro de get_reward model_stimuli_number: {self.model_stimuli_number}')
        if self.current_timestep == self.length_of_experiment - 1:
            if action == self.model_stimuli_number:
                return 1
            return -1
        elif self.current_timestep < 1:
            if action == self.model_stimuli_number:
                self.tact_reward_discount *= .999
                
                if self.tact_reward_discount < 0.01:
                    self.tact_reward_discount = 0
                
                return 1*self.tact_reward_discount
            
        elif action == self.last_action:
            self.autoechoic_reward_discount *= .9999
            if self.autoechoic_reward_discount < 0.01:
                self.autoechoic_reward_discount = 0
            
            return 0.1*self.autoechoic_reward_discount
        
        return -1
           
    def update(self, state, action):
        #new_state = state.copy()
        done = False
        if self.current_timestep >= self.length_of_experiment - 1:
            done = True 
            reward = self.get_reward(action)
            return None, reward, done

        new_state = state.copy()

        #TODO: Nunca se ejecutará este codigo
        #if self.current_timestep == 0:
            
         #   self.model_stimuli_poss = self.spawn_model_stimuli()
          #  new_state[self.model_stimuli_poss[0], self.model_stimuli_poss[1]] = self.model_stimuli_number

        if self.model_steps - 1 <= self.current_timestep < self.length_of_experiment - 2 - self.comparative_steps:
            self.model_stimuli_pos = None
            self.is_model_present = False
            new_state = np.zeros((self.rows, self.cols))
        elif self.length_of_experiment - 2 - self.comparative_steps < self.current_timestep:
            new_state = np.zeros((self.rows, self.cols))
            #print('ENTRO A COMPARATIVE')
            self.comparative_stimuli_poss = np.array(self.spawn_comparative_stimuli())
            new_state[self.comparative_stimuli_poss[:, 0], self.comparative_stimuli_poss[:, 1]] = self.comparative_stimuli_numbers
     
        
        
        # else:
        #     self.restart()
        reward = self.get_reward(action)

        
        # print('new_state')
        # print(new_state)
        
        # print('current_timestamp')
        # print(self.current_timestep)
        return new_state, reward, done
        
    def get_grid(self):
        grid = np.zeros((self.rows, self.cols))
        
        for k, v in self.items_in_grid.items():
            
            pos = np.array(self.items_in_grid[k][0])

            try:
                if len(pos.shape) == 1:
                    pos = pos[np.newaxis, :]

                grid[pos[:, 0], pos[:, 1]] = self.items_in_grid[k][1]
            except:
                pass

        # print('desde get_grid')
        # print(grid)
        return grid
    
    def restart(self):
        self.last_action = -1
        self.current_action = None
        self.current_timestep = 0
        self.model_stimuli_number = np.random.choice(self.comparative_stimuli_numbers)
        # print('self.model_stimuli_number')
        # print(self.model_stimuli_number)
        
        self.is_model_present = True
        self.model_stimuli_pos = self.spawn_model_stimuli()
        self.items_in_grid['model_stimuli'][0] = self.model_stimuli_pos
        self.items_in_grid['model_stimuli'][1] = self.model_stimuli_number
        # print('desde restart')
        # print(self.model_stimuli_pos)
        self.comparative_stimuli_numbers = [s for s in range(1, self.n_stimuli + 1)]
        np.random.shuffle(self.comparative_stimuli_numbers)
        self.done = False
        
        
        
        
class EnvWithStimuli(MinEnv):
    def __init__(self, rows, cols, max_n_blocks=1):
        super().__init__(rows, cols, max_n_blocks)
        
        self.dodge_program = 2
        self.catch_program = -2
        self.current_program_pos = (0, self.cols//2)
        self.current_program_number = self.set_current_program()
        print('self.current_program_number')
        print(self.current_program_number)
        
        self.items_in_grid = {
                                'agent':[self.agent_pos, self.agent_number],
                                'blocks':[self.blocks_pos, self.block_number],
                                'current_program':[self.current_program_pos, self.current_program_number]
                             
                             }
        
    
    def set_current_program(self):
        current_program = np.random.choice([self.catch_program, self.dodge_program])
        return current_program
    
    def get_reward(self, action):
        #todo: cambiar a feature function
        
        if (np.array(self.agent_pos) == self.blocks_pos).all(-1).any():
            if self.current_program_number == self.dodge_program:
                return -1
            elif self.current_program_number == self.catch_program:
                return 1
        return +0.1

    def get_grid(self):
        grid = np.zeros((self.rows, self.cols))
        
        for k, v in self.items_in_grid.items():
            
            pos = np.array(self.items_in_grid[k][0])
            try:
                if len(pos.shape) == 1:
                    pos = pos[np.newaxis, :]
#                 print(k)
#                 print(pos)
                grid[pos[:, 0], pos[:, 1]] = self.items_in_grid[k][1]
            except:
                pass

        #blocks_pos = np.array(self.blocks_pos)

        #grid[blocks_pos[:, 0], blocks_pos[:, 1]] = self.block_number
        
        return grid
    
        
class BaseGridWorld:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.agent_pos = (self.rows - 1, 0)#(rows//2, cols//2)
        self.terminal_states = {'positive':[(0, cols - 1 )], 'negative':[(1, cols - 1)]}
        self.grid = np.zeros((rows, cols))
    
    def update(self, s, a):
        
        r, c = s
        if a == 0:
            if r > 0 :
                r -= 1
        elif a == 1:
            if c < self.cols - 1:
                c += 1
        elif a == 2:
            if r < self.rows - 1:
                r += 1
        elif a == 3:
            if c > 0:
                c -= 1
        
        self.agent_pos = (r, c)
        
        done = False
        
        if (r, c) in self.terminal_states['positive']:
            done = True
            return self.agent_pos, 1, done
        elif (r, c) in self.terminal_states['negative']:
            done = True
            return self.agent_pos, -1, done
        else:
            return self.agent_pos, -0.01, done
    
    def restart(self):
        self.agent_pos = (self.rows - 1, 0)#(self.rows//2, self.cols//2)
    
    def get_grid(self):
        grid = np.zeros((self.rows, self.cols))
        grid[self.agent_pos[0], self.agent_pos[1]] = 1
        grid[self.terminal_states['positive'][0]] = 0.7
        grid[self.terminal_states['negative'][0]] = 0.3
        
        return grid
    
    def get_grid_for_color_render(self):
        grid = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)

        grid[self.agent_pos[0], self.agent_pos[1]] = [255, 255, 0]
        
        #grid[self.agent_pos[0], self.agent_pos[1]] = [255, 255, 0]
        grid[self.terminal_states['positive'][0]] = [0, 255, 0]
        grid[self.terminal_states['negative'][0]] = [255, 0, 0]

        #blocks_pos = np.array(self.blocks_pos)

        #grid[blocks_pos[:, 0], blocks_pos[:, 1]] = [255, 255, 0]
        
        return grid
        
        
        