import numpy as np
from collections import defaultdict
from connect4 import Connect4
import tqdm
from utils import argmax


class QLearningAgent:


    def __init__(self, action_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.state_to_int = {}
        self.state_max = 0


    def value_to_int(self,val):
        tuple_val = tuple(val.flatten())
        if tuple_val in self.state_to_int:
            return self.state_to_int[tuple_val]
        else:
            self.state_max += 1
            self.state_to_int[tuple_val]=self.state_max
            return self.state_max
        

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(self.action_size)
        else:

            return argmax(self.q_table[self.value_to_int(state)])


    def update_q_table(self, state, action, reward, next_state):
        old_q_value = self.q_table[self.value_to_int(state)][action]
        temporal_difference = reward + self.discount_factor * np.max(self.q_table[self.value_to_int(next_state)]) - old_q_value
        new_q_value = old_q_value + self.learning_rate * temporal_difference
        self.q_table[self.value_to_int(state)][action] = new_q_value
