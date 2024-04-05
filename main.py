import argparse
import tqdm
from environment.connect4 import Connect4
from agents.deep_Q import Deep_Q_agent
from agents.Q_learning import QLearningAgent
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from environment.graphics import display_game
import copy
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--ag1", type=str, default="DeepQ", help="agent 1 type (Q_learning, human, random, DeepQ)")
parser.add_argument("--ag2", type=str, default="human", help="agent 2 type (Q_learning, human, random, DeepQ)")
parser.add_argument("--train", type=str, default='True', help="train agents or not")
parser.add_argument("--t_adv", type=str, default='False', help="train against advers")
parser.add_argument("--n", type=int, default=2000, help="number of training episodes")
args = parser.parse_args()

def train_agents(agent1, agent2, n_episodes, deep_agent=[]):
    """ Train two agents against each other
    Args:
        agent1 : first agent
        agent2 : second agent
        n_episodes : number of episodes
        deep_agent : list of the agents that are deep.
    """
    
    # Boucle principale
    for episode in tqdm.tqdm(range(n_episodes)):
        game = Connect4()
        state = game.board
        done = False
        
        while not done:
            # Joueur 1 choisit une action
            action1 = agent1.choose_action(state)
            # Mise à jour de l'état
            next_state, reward, done = game.push(action1, color=0)
            
            if 1 in deep_agent:
                # Mise à jour de la mémoire du joueur 1
                agent1.memorize(copy.deepcopy(state), action1, reward, copy.deepcopy(next_state[:,:,[1,0]]), done)
            else:
                # Mise à jour de la table Q du joueur 1
                agent1.update_q_table(state, action1, reward, next_state)

            state = next_state
            if done:
                break
            # Joueur 2 choisit une action
            action2 = agent2.choose_action(state[:,:,[1,0]])
            # Mise à jour de l'état
            next_state, reward, done = game.push(action2, color=1)
            
            if 2 in deep_agent:
                # Mise à jour de la mémoire du joueur 2
                agent2.memorize(copy.deepcopy(state[:,:,[1,0]]), action2, reward, copy.deepcopy(next_state), done)
            else:
                # Mise à jour de la table Q du joueur 2
                agent2.update_q_table(state, action2, reward, next_state)

            state = next_state
        
        if episode % 10 == 0:
            if 1 in deep_agent:
                if episode % (n_episodes//5) == 0: 
                    agent1.update_q_table(show=True)
                    agent1.exploration_rate -= 0.1
                    print("Agent 1 exploration rate", agent1.exploration_rate)
                else:
                    agent1.update_q_table(show=False)
                
            if 2 in deep_agent:
                if episode % (n_episodes//5) == 0: 
                    agent2.update_q_table(show=True)
                    agent2.exploration_rate -= 0.1
                    print("Agent 2 exploration rate", agent2.exploration_rate)
                else:
                    agent2.update_q_table(show=False)
    return agent1, agent2

if __name__=="__main__":
    deep_agent=[]
    if args.ag1 == "Q_learning":
        ag1 = QLearningAgent(name='Q_learning1')
    elif args.ag1 == "human":
        ag1 = HumanAgent(name='Human1')
    elif args.ag1 == "random":
        ag1 = RandomAgent(name='Random1')
    elif args.ag1 == "DeepQ":
        ag1 = Deep_Q_agent(name='Deep_Q1')
        deep_agent.append(1)
        
    if args.ag2 == "Q_learning":
        ag2 = QLearningAgent(name='Q_learning2')
    elif args.ag2 == "human":
        ag2 = HumanAgent(name='Human2')
    elif args.ag2 == "random":
        ag2 = RandomAgent(name='Random2')
    elif args.ag2 == "DeepQ":
        ag2 = Deep_Q_agent(name='Deep_Q2',n_player=1)
        deep_agent.append(2)
    
    if not (ast.literal_eval(args.train) is False):
        if not (ast.literal_eval(args.t_adv) is False):
            if args.ag1 in ['Q_learning','DeepQ'] and args.ag2 in ['Q_learning','DeepQ']:
                print("Training agent1 against agent2...")
                ag1, ag2 = train_agents(ag1, ag2, args.n, deep_agent)
                print("Training done")
        else:
            if args.ag1 in ['Q_learning','DeepQ']:
                temp = []
                if 1 in deep_agent:
                    temp.append(1)
                print("Training agent1 against random agent...")
                ag1, _ = train_agents(ag1, RandomAgent(), args.n, temp)
            if args.ag2 in ['Q_learning','DeepQ']:
                temp = []
                if 2 in deep_agent:
                    temp.append(2)
                print("Training agent2 against random agent...")
                _, ag2 = train_agents(RandomAgent(), ag2, args.n, temp)
                print("Training done")
    
    display_game(ag1, ag2)
    
