import cv2 

def render_grid(grid):
    
    res = cv2.resize(grid, (200, 200), interpolation=cv2.INTER_NEAREST)
    #print('render grid')
    #print(grid)
    cv2.imshow('ww', res)
    cv2.waitKey(1)

def render_color_grid(grid):
    grid = np.array(grid, dtype=np.uint8)
    res = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
    res = cv2.resize(grid, (200, 200), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('color', res)
    cv2.waitKey(1)
    
def render_q_grid(env, Q):
    grid = np.zeros((env.rows*3, env.cols*3)) 
    values_idxs = np.array([0, 3, 1, 2])
    row_idxs = np.array([0, 1, 1, 2])
    col_idxs = np.array([1, 0, 2, 1])
    Q_r = 0
    Q_c = 0
    for row in range(0, grid.shape[0], 3):
        for col in range(0, grid.shape[1], 3):
            grid[row + row_idxs, col + col_idxs] = Q[Q_r, Q_c, [values_idxs]]
#             print('Q')
#             print(Q[Q_r, Q_c, [values_idxs]])
#             print('Q grid')
#             print(grid[row:row+3, col:col+3])
            Q_c += 1
        Q_r += 1
        Q_c = 0
    
    new_grid = cv2.resize(grid, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Q values', new_grid)
    cv2.waitKey(1)

def render_color_q_grid(env, Q):
    grid = np.zeros((env.rows*3, env.cols*3, 3), np.uint8) 
    values_idxs = np.array([0, 3, 1, 2])
    row_idxs = np.array([0, 1, 1, 2])
    col_idxs = np.array([1, 0, 2, 1])
    Q_r = 0
    Q_c = 0
    reference = np.linspace(-1, 1, 20)
    for row in range(0, grid.shape[0], 3):
        for col in range(0, grid.shape[1], 3):
            q_values = Q[Q_r, Q_c, [values_idxs]]
#             print('q_values')
#             print(q_values)
            new_q_values = reference[np.argmin(np.abs(reference[:,np.newaxis] - q_values), axis=0)][np.newaxis,:]
#             print('new_q_values')
#             print(new_q_values)
            colored_q_values = np.clip(new_q_values.T*[0, 255, -1555], 0, 255)
#             print('colored_q_values')
#             print(colored_q_values)
            grid[row + row_idxs, col + col_idxs] = colored_q_values
#             print('Q')
#             print(Q[Q_r, Q_c, [values_idxs]])
#             print('Q grid')
#             print(grid[row:row+3, col:col+3])
            Q_c += 1
        Q_r += 1
        Q_c = 0
    res = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
    new_grid = cv2.resize(grid, (400, 400), interpolation=cv2.INTER_NEAREST)
    #cv2.imshow('ww', res)
    #cv2.waitKey(1)
    #new_grid = cv2.resize(grid, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Q values', new_grid)
    cv2.waitKey(1)

def render_v_value(agent):
    grid = np.zeros((2, 2), np.uint8) 
    v_s = 0
    try:
    
        total_n_action_taken = np.sum(agent.actions_counter)
        #for a in agent.n_actions:
        v_s = np.dot(agent.actions_counter/total_n_action_taken, agent.current_qs)
    except:
        pass
    
    grid[:, :] = agent.current_qs[agent.current_action]*255#v_s*255
    
    res = cv2.resize(grid, (200, 200))#, interpolation=cv2.INTER_NEAREST)
    print('V(S)')
    print(v_s)
    print('grid v')
    print(grid)
    cv2.imshow('V(s)', res)
    cv2.waitKey(1)
    
    
    
    