
move = ()


def left_click_move(event):
    global move
    move = (event.x, event.y)


class HumanAgent():
     

    def __init__(self, name="Human"):
        self.name = name
        self.move = ()
    

    def conversion(self, move):
        a, b = move[0], move[1]
        if 25 <= a and a <= 375 and 25 <= b and b <= 325:
            return (a - 25) // 50
        else:
            return "Illegal"


    def choose_action(self, fen):
        fen.bind("<Button-1>", left_click_move)
        self.move = self.conversion(self.move)
        if self.move == "Illegal":
            print("Try again !")
            self.move = self.choose_action(fen)
        else:
            return self.move
            