import numpy as np
from environment import grid_world
from visualize import draw_image

WORLD_SIZE = 4
# left, up, right, down
ACTIONS = {'LEFT':np.array([0, -1]), 'UP':np.array([-1, 0]), 'RIGHT':np.array([0, 1]), 'DOWM':np.array([1, 0])}
ACTION_PROB = 0.25



def evaluate_state_value_by_matrix_inversion(env, discount=1.0):
    WIDTH, HEIGHT = env.size()

    # Reward matrix R
    R = np.zeros((WIDTH, HEIGHT))
    for i in range(WIDTH):
        for j in range(HEIGHT):
            expected_reward = 0
            for action in ACTIONS:
                (next_i, next_j), reward = env.interaction([i, j], ACTIONS[action])
                expected_reward += ACTION_PROB*reward
            R[i, j] = expected_reward
    R = R.reshape((-1,1))
    R = R[1:-1,:]
    print(R)


    # Transition matrix T
    
    
    T = [[0.25,0.25,0,0,0.25,0,0,0,0,0,0,0,0,0],[0.25,0.25,0.25,0,0,0.25,0,0,0,0,0,0,0,0],[0,0.25,0.5,0,0,0,0.25,0,0,0,0,0,0,0],[0,0,0,0.25,0.25,0,0,0.25,0,0,0,0,0,0],[0.25,0,0,0.25,0,0.25,0,0,0.25,0,0,0,0,0],[0,0.25,0,0,0.25,0,0.25,0,0,0.25,0,0,0,0],[0,0,0.25,0,0,0.25,0.25,0,0,0,0.25,0,0,0],[0,0,0,0.25,0,0,0,0.25,0,0,0,0.25,0,0],[0,0,0,0,0.25,0,0,0.25,0,0.25,0,0,0.25,0],[0,0,0,0,0,0.25,0,0,0.25,0,0.25,0,0,0.25],[0,0,0,0,0,0,0.25,0,0,0.25,0.25,0,0,0],[0,0,0,0,0,0,0,0.25,0,0,0,0.5,0.25,0],[0,0,0,0,0,0,0,0,0.25,0,0,0.25,0.25,0.25],[0,0,0,0,0,0,0,0,0,0.25,0,0,0.25,0.25]]
    T = np.array(T)
    
    
    
    
    V = np.zeros((WIDTH, HEIGHT))
    V = V.reshape((-1,1))
    V = V[1:-1,:]
    
    k = 1

    for _ in range(k):
        V = R + T.dot(V)
        
    V = np.insert(V,0,0,axis=0)
    V = np.insert(V,15,0,axis=0)
    


    
    new_state_values = V.reshape(WIDTH,HEIGHT)
    draw_image(k, np.round(new_state_values, decimals=2))

    return new_state_values


if __name__ == '__main__':
    env = grid_world()
    values = evaluate_state_value_by_matrix_inversion(env = env)