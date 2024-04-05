import numpy as np
class RandomAgent():
     
    def __init__(self, n_actions=7,name="Random"):
        self.name = name
        self.move = ()
        self.n_actions = n_actions
        self.n_player=0

    def choose_action(self, state, verbose=False):
        return np.random.randint(0,self.n_actions)
    
    def update_q_table(self, state, action, reward, next_state):
        pass
    