import numpy as np

HEIGHT = 4
WIDTH = 4

class grid_world:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT


    def is_terminal(self, state):   # Gaol state
        x, y = state
        return (x == 0 and y == 0) or (x == self.width - 1 and y == self.height - 1)


    def interaction(self, state, action):
        if self.is_terminal(state):
            return state, 0

        next_state = (np.array(state) + action).tolist()
        x, y = next_state

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            next_state = state

        reward = -1
        return next_state, reward


    def size(self):
        return self.width, self.height
