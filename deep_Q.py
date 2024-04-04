from connect4 import Connect4
import copy
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import random

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def argmax(q_values):

    top = float("-inf")
    ties = []

    for i in range(len(q_values)):

        if q_values[i] > top:
            top = q_values[i]
            ties = [i]
        elif q_values[i] == top:
            ties.append(i)
    ind=np.random.choice(ties)
    return ind

class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Deep_Q_agent:

    def __init__(self, learning_rate=1e-4, discount_factor=0.99, exploration_rate=0.5,n_player=0):
        self.n_player = n_player
        self.action_size = 7
        self.state_size = 7*6*2
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.model = ConvNet(optim.Adam, nn.MSELoss(),lr=learning_rate,disc_f=discount_factor)
        self.model.to(self.model.device)
        self.model.eval()
        self.memory = []

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def clear_memory(self):
        self.memory = []

    def choose_action(self, state,verbose=False):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(self.action_size)
        else:
            if self.n_player==0:
                if verbose:
                    print(self.model(torch.from_numpy(state).float().unsqueeze(0).to(device)))
                return argmax(self.model(torch.from_numpy(state).float().unsqueeze(0).to(device)).detach().cpu().numpy()[0])
            else:
                if verbose:
                    print(self.model(torch.from_numpy(state[:,:,[1,0]]).float().unsqueeze(0).to(device)))
                return argmax(self.model(torch.from_numpy(state[:,:,[1,0]]).float().unsqueeze(0).to(device)).detach().cpu().numpy()[0])


    def update_q_table(self,show=False):
        dataset = CustomDataset(self.memory)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.model.train_model(train_loader,show=show)
        self.clear_memory()

class ConvNet(nn.Module):

    def __init__(self, optimizer=optim.Adam, criterion=nn.MSELoss(),lr=1e-2,disc_f=0.99):
        self.input_shape = (7, 6,2)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=self.input_shape[2], out_channels=32, kernel_size=5, padding=2)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, padding=2)

        self.relu2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=672, out_features=128)
        self.relu3 = nn.ReLU(inplace=True)
        self.output = nn.Linear(128, 7) 
        
        self.optimizer = optimizer(self.parameters(),lr,weight_decay=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.9)
        self.criterion = criterion
        self.discount_factor = disc_f
        self.batch =nn.BatchNorm1d(7)
 
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
      x = x.transpose(1,3)#reshape(x.shape[0], 2, 7, 6)
      x = x.transpose(2,3)
      x = self.relu1(self.conv1(x))
#      x = self.pool1(x)
      x = self.relu2(self.conv2(x))
      x = self.relu2(self.conv3(x))
      x = self.relu2(self.conv4(x))
      x = self.flatten(x)
      x = self.relu3(self.fc1(x))
      output = self.output(x)
      return output#self.softmax(output)

    def train_model(self, train_loader, epochs=1,show=False):
        for p in self.parameters():
          p.requires_grad = True
        self.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, batch_data in enumerate(train_loader):
                states, actions, rewards, next_states, dones = batch_data
                states = states.to(self.device).float()
                actions = actions.to(self.device)
                rewards = rewards.to(self.device).float()
                #next_states = next_states.to(self.device).float()
                dones = dones.to(self.device)
                
                targets = rewards.clone()
                with torch.no_grad():
                   # next_state_values = self(next_states).max(1)[0].detach()
                    targets[~dones] = targets[~dones]-self.discount_factor * next_states[~dones]#next_state_values[~dones]
                targets_f = self(states)
                actions = actions.long()  # Convert actions to long type
                indices = torch.arange(len(targets_f)).long()  # Ensure indices are of type long

                # Use the corrected indices
                targets_f[indices, actions] = targets
                self.optimizer.zero_grad()
                outputs = self(states)
                loss = self.criterion(outputs[indices, actions], targets_f[indices, actions])
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                self.scheduler.step()

            if show : 
                if epoch ==0:
                    print(f"Loss: {running_loss}")
                    print(self.scheduler.get_last_lr())
        self.eval()




example_state = np.array([[[0., 1.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.]],

                  [[1., 0.],
                   [1., 0.],
                   [1., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.]],

                  [[0., 1.],
                   [0., 1.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.]],

                  [[0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.]],

                  [[0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.]],

                  [[0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.]],

                  [[0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.]]])


example_tensor = torch.tensor([example_state],dtype=torch.float)




example_score = []


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer))
        else:
            return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

buffer0=ReplayBuffer(1e7)
buffer1=ReplayBuffer(1e7)

def train_agents(agent1,it,repbuffer0=buffer0,repbuffer1=buffer1,against_random=False):
    with tqdm(total=it) as pbar:
        for episode in range(0,it):
            game=Connect4()
            state = copy.deepcopy(game.board)
            done = False
            while not done:
                state=game.board
                action0 = agent1.choose_action(state)
                next_state0, reward0, done = game.push(action0, color=0)
               
                repbuffer0.add((action0,next_state0, 1.0*reward0, done,state))
                batch = repbuffer0.sample(1)
                action0,next_state0, reward0, done, state = batch[0]
                agent1.memorize(copy.deepcopy(state), action0, reward0,agent1.model(copy.deepcopy(torch.tensor([next_state0[:,:,[1,0]]]).float())).max().detach(), done)

                if not done:
                    state = game.board
                    action0 = agent1.choose_action(state[:,:,[1,0]])
                    next_state0, reward0, done = game.push(action0, color=1)
                    repbuffer1.add((action0,next_state0, 1.0*reward0, done,state))
                    batch = repbuffer1.sample(1)
                    action0,next_state0, reward0, done, state = batch[0]
                    agent1.memorize(copy.deepcopy(state[:,:,[1,0]]), action0, reward0,agent1.model(copy.deepcopy(torch.tensor([next_state0]).float())).max().detach(), done)

        
            if episode % 10 == 0:
                if episode %100 == 0:
                    show=True
                    pbar.update(100)
                    example = agent1.model(example_tensor)
                    example = example-example.mean()
                    example=example/torch.mean(example**2)**0.5
                    print(example[0,1])
                    example_score.append(example[0].detach().numpy())
                else:
                    show=False
                agent1.update_q_table(show=show)
         

train=False
if train:
    agent1 = Deep_Q_agent()

    # agent1.exploration_rate=0.8
    # train_agents(agent1,agent2,it=2000)

    # game = Connect4()
    # game.play_a_game(agent1,agent2)

    agent1.exploration_rate=0.5
    train_agents(agent1,it=200)
    agent2 = copy.deepcopy(agent1)
    agent2.n_player=1

    game = Connect4()
    game.play_a_game(agent1,agent2)

    agent1.exploration_rate=0.3
    train_agents(agent1,it=2000)
    agent2 = copy.deepcopy(agent1)

    agent2.n_player=1

    game = Connect4()
    game.play_a_game(agent1,agent2)
    

    # agent1.exploration_rate=0.2
    # train_agents(agent1,agent2,it=2000)
    agent2 = copy.deepcopy(agent1)

    agent2.n_player=1

    game = Connect4()
    game.play_a_game(agent1,agent2)

    agent1.exploration_rate=0.1
    train_agents(agent1,it=2000)

    agent1.exploration_rate=0.05
    agent2 = copy.deepcopy(agent1)

    game.reset()

    agent2.n_player=1
    np.save('experience.npy',example_score)
    agent1.exploration_rate=0
    agent2.exploration_rate=0
    game.play_a_game(agent1,agent2)

    game = Connect4()
    print(game.play_n_games(agent1,agent2,1000))
    