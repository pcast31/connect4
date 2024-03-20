from connect4 import Connect4
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class Deep_Q_agent:

    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0):
        self.action_size = 7
        self.state_size = 7*6*2
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((7,6,2))
        self.model = ConvNet(optim.Adam, nn.MSELoss())
        self.memory = []

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def clear_memory(self):
        self.memory = []

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(self.action_size)
        else:
            print(state.shape)
            return np.argmax(self.model(state).detach().numpy())
    
    def update_q_table(self):
        self.model.train_model(self.memory)
        self.clear_memory()

class ConvNet(nn.Module):
    
    def __init__(self, optimizer=optim.Adam, criterion=nn.MSELoss()):        
        
        
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 16, kernel_size=4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 7)
        self.optimizer = optimizer(self.parameters())
        self.criterion = criterion
        self.discount_factor = 0.9

    def forward(self, x):
        x= torch.tensor(np.transpose(x, (2, 0, 1))).float()
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train_model(self, train_loader, epochs=5):

        self.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                state, action, reward, next_state, done = data
                target = reward
                if not done:
                    target = reward + self.discount_factor * np.max(self(next_state).detach().numpy())
                target_f = self(state)
                target_f[0][action] = target
                self.optimizer.zero_grad()
                output = self(state)
                loss = self.criterion(output, target_f)
                loss.backward()
                self.optimizer.step()
                
        self.eval()
         
def playgame():
    game = Connect4()
    agent = Deep_Q_agent()
    for episode in range(1000):
        state = game.board
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = game.push(action, color=0)
            
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if episode % 2 == 0:
            agent.update_q_table()
    return agent,agent
