import numpy as np
from collections import defaultdict
from .utils import argmax

class QLearningAgent:
    """Class for Q-learning agent
    """
    def __init__(self, name = "Q-learning",learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.action_size = 7
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(7))
        self.state_to_int = {}
        self.state_max = 0
        self.name = name

    def value_to_int(self, state):
        """Converts a value to an integer (used to index the Q-table)
        Args:
            state (np.array): state/board
        Returns:
            Index
        """
        tuple_val = tuple(state.flatten())
        if tuple_val in self.state_to_int:
            return self.state_to_int[tuple_val]
        else:
            self.state_max += 1
            self.state_to_int[tuple_val] = self.state_max
            return self.state_max

    def choose_action(self, state):
        """Chooses an action based on the current state
        Args:
            state
        Returns:
            action between 0 and 6
        """
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(self.action_size)
        else:
            return argmax(self.q_table[self.value_to_int(state)])

    def update_q_table(self, state, action, reward, next_state):
        """Updates the Q-table
        """
        old_q_value = self.q_table[self.value_to_int(state)][action]
        temporal_difference = reward + self.discount_factor * np.max(self.q_table[self.value_to_int(next_state)]) - old_q_value
        new_q_value = old_q_value + self.learning_rate * temporal_difference
        self.q_table[self.value_to_int(state)][action] = new_q_value