import numpy as np


class RandomAgent():
     

    def __init__(self, name="Random"):
        self.name = name
        self.move = ()


    def choose_action(self, legal_moves):
        move = np.random.choice(np.flatnonzero(legal_moves == legal_moves.max()))
        return(move)
    