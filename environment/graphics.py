from environment.connect4 import Connect4
import tkinter as tk
import tkinter.messagebox
import time


def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)


tk.Canvas.create_circle = _create_circle


def display_board(game):
    #Ligne de séparation :
    game.canva.create_line(400, 0, 400, 350, fill="black", width=3)

    #Quadrillage :
    #Lignes :
    l = [25 + 50*k for k in range(0, 7)]
    for k in l:
        game.canva.create_line(25, k, 375, k, fill="black")
    
    #Colonnes :
    l = [25 + 50*k for k in range(0, 8)]
    for k in l:
        game.canva.create_line(k, 25, k, 325, fill="black")    
    
    #Placement des pions :
    bo = game.board
    r = 20
    for x in range(7):
        for y in range(6):
            if bo[x][y][0] == 1:
                game.canva.create_circle(50 + 50*x, 50 + 50*(5-y), r, fill='red')
            if bo[x][y][1] == 1:
                game.canva.create_circle(50 + 50*x, 50 + 50*(5-y), r, fill='gold')                

    #initiative
    if game.k % 2 == 1:
        game.canva.create_circle(425, 25, 5, fill='red')
    if game.k % 2 == 0:
        game.canva.create_circle(425, 25, 5, fill='gold')
    
    return


move = -1

def left_click_move(event):
    global move
    move = (event.x, event.y)


def make_a_move(game, player):
    global move

    if "Human" not in player.name:
        time.sleep(0.5)
        move = player.choose_action(game.board)
    else:
        move = player.choose_action(game)
    return move


def endgame(game,player1_n,player2_n):
    print("Game ended")
    display_board(game)
    game.canva.update()
    if game.draw:
        print("Draw !")
        tk.messagebox.showinfo('Game ended', 'Draw !')
        return 0
    elif game.k % 2 == 0 :
        print("Red won !" )
        tk.messagebox.showinfo('Game ended', f'{player1_n} won !')
        return 1
    elif game.k % 2 == 1 :
        print("Yellow won !" )
        tk.messagebox.showinfo('Game ended', f'{player2_n} won!')
        return -1


def play_game(game, player1, player2):
    global move

    # Player 1 move
    if (game.k + 1) % 2 == 0:
        move = make_a_move(game, player1)

    # Player 2 move
    else:
        move = make_a_move(game, player2)

    # Make the input move
    if move != -1:
        game.push(move, (game.k + 1) % 2)
        game.canva.delete('all')
        display_board(game)
        game.canva.update()
        move = -1

    # Check if game is over
    if game.game_over:
        if game.illegal_move:
            tk.messagebox.showinfo('Game ended', 'Illegal move !')
            return(endgame(game,player1.name,player2.name))
        else:
            return(endgame(game,player1.name,player2.name))
        
    game.canva.after(20, play_game, game, player1, player2)


def display_game(player1, player2):
    """
    Starts main function
    """
    wid = tk.Tk(baseName="Connect4 " + player1.name + " vs " + player2.name)
    can = tk.Canvas(wid, bg='light grey', height=350, width=450) 
    can.pack()
    game = Connect4()
    game.window = wid
    game.canva = can
    display_board(game)
    can.update()
    play_game(game, player1, player2)
    wid.mainloop()
