import argparse
import tqdm
from connect4 import Connect4
from Q_learning import QLearningAgent
from human_agent import HumanAgent
from random_agent import RandomAgent
from graphics import display_game


parser = argparse.ArgumentParser()
# parser.add_argument("--agent_1", type=Agent, help="class of the first agent")
# parser.add_argument("--agent_2", type=Agent, help="class of the second agent")
parser.add_argument("--n", type=int, default=1000, help="number of training episodes")
parser.add_argument("--a", type=int, default=7, help="Q-learning agent action size")
args = parser.parse_args()


def train_agents(agent1, agent2, n_episodes):
    # Boucle principale
    for episode in tqdm.tqdm(range(n_episodes)):
        game = Connect4()
        state = game.board
        done = False
        
        while not done:
            # Joueur 1 choisit une action
            action1 = agent1.choose_action(game)
            # Mise à jour de l'état
            next_state, reward, done = game.push(action1, color=0)
            # Mise à jour de la table Q du joueur 1
            agent1.update_q_table(state, action1, reward, next_state)

            state = next_state
            if done:
                break

            # Joueur 2 choisit une action
            action2 = agent2.choose_action(game)
            # Mise à jour de l'état
            next_state, reward, done = game.push(action2, color=1)
            # Mise à jour de la table Q du joueur 2
            agent2.update_q_table(state, action2, reward, next_state)

            state = next_state
    return agent1, agent2


if __name__=="__main__":
    game = Connect4()
    ag1, ag2 = QLearningAgent(action_size=args.a, name="QAgent1"), QLearningAgent(action_size=args.a, name="QAgent2")
    ag1, ag2 = train_agents(ag1, ag2, args.n)
    hum1, hum2 = HumanAgent(), HumanAgent()
    rand1, rand2 = RandomAgent(), RandomAgent()
    # winner = game.play_a_game(ag1, hum1, show_game=True)
    display_game(hum1, ag2)
    # print(game.play_n_games(ag1, rand1, 5000))
