import argparse
import tqdm
from connect4 import Connect4
from Q_learning import QLearningAgent
from human_player import HumanAgent
from graphics import display_game


parser = argparse.ArgumentParser()
# parser.add_argument("--agent_1", type=Agent, help="class of the first agent")
# parser.add_argument("--agent_2", type=Agent, help="class of the second agent")
parser.add_argument("--n_episodes", type=int, default=1000, help="number of training episodes")
parser.add_argument("--action_size", type=int, default=7, help="Q-learning agent action size")
args = parser.parse_args()


def train_agents(agent1, agent2, n_episodes):
    # Boucle principale
    for episode in tqdm.tqdm(range(n_episodes)):
        game = Connect4()
        state = game.board
        done = False
       # print(episode)
        while not done:
            # Joueur 1 choisit une action
            action1 = agent1.choose_action(state)
            # Mise à jour de l'état
            next_state, reward, done = game.push(action1, color=0)
            # Mise à jour de la table Q du joueur 1
            agent1.update_q_table(state, action1, reward, next_state)

            state = next_state
            if done:
                break

            # Joueur 2 choisit une action
            action2 = agent2.choose_action(state)
            # Mise à jour de l'état
            next_state, reward, done = game.push(action2, color=1)
            # Mise à jour de la table Q du joueur 2
            agent2.update_q_table(state, action2, reward, next_state)

            state = next_state
    return agent1, agent2


if __name__=="__main__":
    game = Connect4()
    ag1, ag2 = QLearningAgent(action_size=args.action_size, name="QAgent1"), QLearningAgent(action_size=args.action_size, name="QAgent2")
    ag1, ag2 = train_agents(ag1, ag2, args.n_episodes)
    hum1, hum2 = HumanAgent(), HumanAgent()
    # game.play_a_game(ag1, ag2)
    display_game(hum1, ag2)
