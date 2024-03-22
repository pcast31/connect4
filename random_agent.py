import numpy as np


class RandomAgent():
     

    def __init__(self, name="Random"):
        self.name = name


    def choose_action(self, game):
        move = np.random.choice(np.flatnonzero(game.legal_moves == game.legal_moves.max()))
        return(move)
    