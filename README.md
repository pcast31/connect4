# Project Title

Reinforcement Learning project : Connect4 using Deep-Q Learning

Students : Arthur Walker, François Wang, Paul Castéras


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

This project requires Python 3.8 (3.8.18 works). You can install the required packages by running the following command:

```bash 
pip install -r requirements.txt
```

# Usage

To play the game, run the following command:

```bash
python main.py
```
The game will start and you will be able to play against the AI. The AI is trained using Deep-Q Learning. The AI will play first and you will play second. You can play by entering the column number where you want to play. The columns are numbered from 0 to 6. 

You can modify the default parameters :

- Parameter --ag1 : the first agent which can be chosen from ('Q_learning', 'human', 'random', 'DeepQ') (default is 'DeepQ'). ```Exemple : python main.py --agent1 Q_learning```
- Parameter --ag2 : the second agent which can be chosen from ('Q_learning', 'human', 'random', 'DeepQ') (default is 'human'). ```Exemple : python main.py --agent2 random```
- Parameter --train : if one of the agents is a DeepQ or Q_learning agent, you can choose to train it or not. (default is True). ```Exemple : python main.py --train False```
- Parameter --n : if train is True, you can choose the number of episodes for the training (default is 2000). ```Exemple : python main.py --n 20000```
- Parameter --t_adv : if train is True, and if both agents are trainable, you can choose to train them against each other or against a random agent (default is False). ```Exemple : python main.py --t_adv True```

To see the help, run the following command:

```bash
python main.py --help
```

Exemple : 

```bash
python main.py --agent1 human --agent2 random --train False --n 20000
```

