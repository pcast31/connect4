from connect4 import Connect4
import copy
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm

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


class Deep_Q_agent:

    def __init__(self, learning_rate=10**-3, discount_factor=0.5, exploration_rate=0.5,n_player=0):
        self.n_player = n_player
        self.action_size = 7
        self.state_size = 7*6*2
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((7,6,2))
        self.model = ConvNet(optim.Adam, nn.MSELoss(),lr=learning_rate,disc_f=discount_factor)
        self.model.to(self.model.device)
        self.model.eval()
        self.memory = []

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def clear_memory(self):
        self.memory = []

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(self.action_size)
        else:
            if self.n_player==0:
                return argmax(self.model(torch.from_numpy(state).float().unsqueeze(0).to(device)).detach().cpu().numpy()[0])
            else:
                return argmax(self.model(torch.from_numpy(state[:,:,[1,0]]).float().unsqueeze(0).to(device)).detach().cpu().numpy()[0])


    def update_q_table(self,show=False):
        dataset = CustomDataset(self.memory)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.model.train_model(train_loader,show=show)
        self.clear_memory()

class ConvNet(nn.Module):

    def __init__(self, optimizer=optim.Adam, criterion=nn.MSELoss(),lr=10**-5,disc_f=0.5):
        self.input_shape = (7, 6,2)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=self.input_shape[2], out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=576, out_features=128)
        self.relu3 = nn.ReLU(inplace=True)
        self.output = nn.Linear(128, 7) 
        
        self.optimizer = optimizer(self.parameters(),lr)
        self.criterion = criterion
        self.discount_factor = disc_f

    def forward(self, x):
      x = x.reshape(x.shape[0], 2, 7, 6)
      x = self.relu1(self.conv1(x))
      x = self.pool1(x)
      x = self.relu2(self.conv2(x))
      x = self.flatten(x)
      x = self.relu3(self.fc1(x))
      output = self.output(x)
      return output

    def train_model(self, train_loader, epochs=10,show=False):
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
                next_states = next_states.to(self.device).float()
                dones = dones.to(self.device)
                
                targets = rewards.clone()
                with torch.no_grad():
                    next_state_values = self(next_states).max(1)[0].detach()
                    targets[~dones] += self.discount_factor * next_state_values[~dones]
                targets_f = self(states)
                targets_f[range(len(targets_f)), actions] = targets
                self.optimizer.zero_grad()
                outputs = self(states)
                loss = self.criterion(outputs, targets_f)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

            if show : 
                if epoch ==0:
                    print(f"Loss: {running_loss}")
        self.eval()

def train_agents(agent,it):
    with tqdm(total=it) as pbar:
        for episode in range(1,it):
            game=Connect4()
            state = copy.deepcopy(game.board)
            done = False
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done = game.push(action, color=0)
                if reward <0:
                    reward=reward*1
                agent.memorize(copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done)
                state = next_state
                if not done :
                    opp_state = state[:,:,[1,0]]
                    action = agent.choose_action(opp_state)
                    next_state, reward, done = game.push(action, color=1)
                    if reward <0:
                        reward=reward*1
                    opp_next_state = next_state[:,:,[1,0]]
                    agent.memorize(opp_state, action, reward, opp_next_state, done)
                    state = next_state
            if episode % 10 == 0:
                if episode %100 == 0:
                    show=True
                    pbar.update(100)
                else:
                    show=False
                agent.update_q_table(show=show)
         

# train=False
# if train:
#     agent = Deep_Q_agent()
#     train_agents(agent,it=20000)
#     agent.exploration_rate=0.4
#     train_agents(agent,it=20000)
#     agent.exploration_rate=0.3
#     train_agents(agent,it=20000)
#     agent.exploration_rate=0.2
#     train_agents(agent,it=20000)
#     agent.exploration_rate=0.1
#     train_agents(agent,it=20000)
#     agent.exploration_rate=0
#     agent2=copy.deepcopy(agent)
#     agent2.n_player=1