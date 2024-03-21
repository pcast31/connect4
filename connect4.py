import numpy as np


#----------------------------------------------------------------------------------------------------------     
#Connect four
class Connect4():


    def __init__(self):
        super().__init__()
        self.board = np.zeros((7, 6, 2))
        self.str = "0" * 7 * 6 * 2 + "0"
        self.legal_moves = np.ones(7)
        self.cellHeight = np.zeros(7)
        self.representation = self.board
        self.allMove = [i for i in range(7)]
        self.k = 1
        self.winner = 0
        self.draw = False
        self.game_over = False

    
    def reset(self):
        return


    def push(self, pos, color=0):
        if np.sum(self.board[pos, 5])>0:
        #    self.show()
        #    print(self.legal_moves)
         #   print(pos)
            #self.playRandomMove()
            return self.board, -1, True

        self.board[pos, int(self.cellHeight[pos]), color] = 1
        self.cellHeight[pos] += 1
        if self.cellHeight[pos] > 5:
            self.legal_moves[pos] = 0

        pos = [pos, int(self.cellHeight[pos]) - 1]

        win = False

        #check left
        if pos[0] >= 3:
            if 1==self.board[pos[0]-3,pos[1],color]==self.board[pos[0]-2,pos[1],color]==self.board[pos[0]-1,pos[1],color]:
                win=True
        if pos[0] >= 2 and pos[0] < 6:
            if 1==self.board[pos[0]-2,pos[1],color]==self.board[pos[0]-1,pos[1],color]==self.board[pos[0]+1,pos[1],color]:
                win=True

        #check bottom left
        if pos[0] >= 3 and pos[1] >= 3:
            if 1==self.board[pos[0]-3,pos[1]-3,color]==self.board[pos[0]-2,pos[1]-2,color]==self.board[pos[0]-1,pos[1]-1,color]:
                win=True
        if pos[0] >= 2 and pos[1] >= 2 and pos[0] < 6 and pos[1] < 5:
            if 1==self.board[pos[0]-2,pos[1]-2,color]==self.board[pos[0]-1,pos[1]-1,color]==self.board[pos[0]+1,pos[1]+1,color]:
                win=True
    
        #check top left
        if pos[0] >= 3 and pos[1] <= 2:
            if 1==self.board[pos[0]-3,pos[1]+3,color]==self.board[pos[0]-2,pos[1]+2,color]==self.board[pos[0]-1,pos[1]+1,color]:
                win=True
        if pos[0] >= 2 and pos[1] <= 3 and pos[0] < 6 and pos[1] > 0:
            if 1==self.board[pos[0]-2,pos[1]+2,color]==self.board[pos[0]-1,pos[1]+1,color]==self.board[pos[0]+1,pos[1]-1,color]:
                win=True
    
        #check right
        if pos[0] <= 3:
            if 1==self.board[pos[0]+3,pos[1],color]==self.board[pos[0]+2,pos[1],color]==self.board[pos[0]+1,pos[1],color]:
                win=True
        if pos[0] <= 4 and pos[0] > 0:
            if 1==self.board[pos[0]+2,pos[1],color]==self.board[pos[0]+1,pos[1],color]==self.board[pos[0]-1,pos[1],color]:
                win=True    
                
        #check bottom right
        if pos[0] <= 3 and pos[1] >= 3:
            if 1==self.board[pos[0]+3,pos[1]-3,color]==self.board[pos[0]+2,pos[1]-2,color]==self.board[pos[0]+1,pos[1]-1,color]:
                win=True
        if pos[0] <= 4 and pos[1] >= 2 and pos[0] > 0 and pos[1] < 5:
            if 1==self.board[pos[0]+2,pos[1]-2,color]==self.board[pos[0]+1,pos[1]-1,color]==self.board[pos[0]-1,pos[1]+1,color]:
                win=True
            
        #check top right
        if pos[0] <= 3 and pos[1] <= 2:
            if 1==self.board[pos[0]+3,pos[1]+3,color]==self.board[pos[0]+2,pos[1]+2,color]==self.board[pos[0]+1,pos[1]+1,color]:
                win=True
        if pos[0] <= 4 and pos[1] <= 3 and pos[0] > 0 and pos[1] > 0:
            if 1==self.board[pos[0]+2,pos[1]+2,color]==self.board[pos[0]+1,pos[1]+1,color]==self.board[pos[0]-1,pos[1]-1,color]:
                win=True

        #check bottom
        if pos[1] >= 3:
            if 1==self.board[pos[0],pos[1]-3,color]==self.board[pos[0],pos[1]-2,color]==self.board[pos[0],pos[1]-1,color]:
                win = True

        if win:
            self.game_over = True
            self.winner = (1, -1)[color]

        self.str = self.str[:pos[0]*7 + pos[1]] + str(self.k) + self.str[pos[0]*7 + pos[1] + 1: -1] + str(pos[0])
        self.k += 1
        if np.max(self.legal_moves) == 0 and self.winner == 0:
            self.game_over = True
            self.draw = True
        self.representation = self.board
    
        return self.board, win, self.game_over


    def get_legal_moves(self):
        moves = []
        for i in range(7):
            if self.legal_moves[i] == 1:
                moves.append(i)
        return moves


    def playRandomMove(self):
        p = np.random.choice(np.flatnonzero(self.legal_moves == self.legal_moves.max()))
        self.push(p)


    def show(self):
        print("-------")
        for a in range(5, -1, -1):
            line = ""
            for b in range(7):
                if self.board[b, a, 0] == 1:
                    line += "x"
                elif self.board[b, a, 1] == 1:
                    line += "o"
                else:
                    line += " "
            print(line)
        print("-------")
    

    def play_a_game(self, agent1, agent2):
        state = self.board
        done = False
        while not done:
            # Joueur 1 choisit une action
            action1 = agent1.choose_action(state)
            # Mise à jour de l'état
            next_state, reward, done = self.push(action1, color=0)
            # Mise à jour de la table Q du joueur 1
            state = next_state
            self.show()
            if done:
                break

            # Joueur 2 choisit une action
            action2 = agent2.choose_action(state)
            # Mise à jour de l'état
            next_state, reward, done = self.push(action2,color=1)
            # Mise à jour de la table Q du joueur 2
            state = next_state
            self.show()
