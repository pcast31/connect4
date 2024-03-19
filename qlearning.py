import numpy as np
from connect4 import Connect4
from collections import defaultdict
import tqdm






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

            return np.argmax(self.q_table[self.value_to_int(state)])

    def update_q_table(self, state, action, reward, next_state):
        old_q_value = self.q_table[self.value_to_int(state)][action]
        temporal_difference = reward + self.discount_factor * np.max(self.q_table[self.value_to_int(next_state)]) - old_q_value
        new_q_value = old_q_value + self.learning_rate * temporal_difference
        self.q_table[self.value_to_int(state)][action] = new_q_value


def play_game():
    # Création des agents Q-learning
    agent1 = QLearningAgent(action_size=7)
    # Boucle principale
    for episode in tqdm.tqdm(range(100000)):
        game = Connect4()
        state = game.board
        done = False
       # print(episode)
        while not done:
            # Joueur 1 choisit une action
            action1 = agent1.choose_action(state)
            # Mise à jour de l'état
            next_state, reward, done = game.push(action1,color=0)
            # Mise à jour de la table Q du joueur 1
            agent1.update_q_table(state, action1, reward, next_state)

            state = next_state
            if done:
                break

            # Joueur 2 choisit une action
            action2 = agent1.choose_action(state)
            # Mise à jour de l'état
            next_state, reward, done = game.push(action2,color=1)
            # Mise à jour de la table Q du joueur 2
            agent1.update_q_table(state, action2, reward, next_state)

            state = next_state
    return agent1


def play_a_game(agent1,agent2):
    game = Connect4()
    state = game.board
    done=False
    while not done:
            # Joueur 1 choisit une action
            action1 = agent1.choose_action(state)
            # Mise à jour de l'état
            next_state, reward, done = game.push(action1,color=0)
            # Mise à jour de la table Q du joueur 1

            if done:
                break
            state = next_state
            # Joueur 2 choisit une action
            action2 = agent2.choose_action(state)
            # Mise à jour de l'état
            next_state, reward, done = game.push(action2,color=1)
            # Mise à jour de la table Q du joueur 2
            state = next_state
            game.show()

if __name__=="__main__":
    ag1 = play_game()
    play_a_game(ag1,ag1)