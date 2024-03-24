

class HumanAgent():
     

    def __init__(self, name="Human"):
        self.name = name
        self.move = -1
    

    def conversion(self, move):
        a, b = move[0], move[1]
        if 25 <= a and a <= 375 and 25 <= b and b <= 325:
            return (a - 25) // 50
        else:
            return "Illegal"
    

    def left_click_move(self, event):
        self.move = (event.x, event.y)


    def choose_action(self, game):
        if game.window == None:
            move = int(input())
        else:
            move = -1
            game.window.bind("<Button-1>", self.left_click_move)
            if self.move != -1:
                self.move = self.conversion(self.move)
                if self.move == "Illegal":
                    print("Missclick !")
                    move = -1
                    self.move = -1
                else:
                    move = self.move
                    self.move = -1
        return(move)
