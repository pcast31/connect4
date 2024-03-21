from connect4 import Connect4
import tkinter as tk
import tkinter.messagebox
import time
import numpy as np


def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)


tk.Canvas.create_circle = _create_circle


def display_board(game, can):
    #Ligne de séparation :
    can.create_line(400, 0, 400, 350, fill="black", width=3)

    #Quadrillage :
    #Lignes :
    l = [25 + 50*k for k in range(0, 7)]
    for k in l:
        can.create_line(25, k, 375, k, fill="black")
    
    #Colonnes :
    l = [25 + 50*k for k in range(0, 8)]
    for k in l:
        can.create_line(k, 25, k, 325, fill="black")    
    
    #Placement des pions :
    bo = game.board
    r = 20
    for x in range(7):
        for y in range(6):
            if bo[x][y][0] == 1:
                can.create_circle(50 + 50*x, 50 + 50*(5-y), r, fill='red')
            if bo[x][y][1] == 1:
                can.create_circle(50 + 50*x, 50 + 50*(5-y), r, fill='gold')                

    #initiative
    if game.k % 2 == 1:
        can.create_circle(425, 25, 5, fill='red')
    if game.k % 2 == 0:
        can.create_circle(425, 25, 5, fill='gold')
    
    return


move = ()

def left_click_move(event):
    global move
    move = (event.x, event.y)


def make_a_move(game, player):
    if "Human" in player.name:
        fen.bind("<Button-1>", left_click_move)
        if move != ():
            print(move)
            move = player.conversion(move)
            if move == "Illegal":
                print("Try again !")
                move = make_a_move(game, player)
    else:
        state = game.board
        move = player.choose_action(state)
    return(move)


def endgame(game):
    if game.game_over :
        print("Game ended")
        display_board(game, can)
        can.update()
        time.sleep(0.5)
        if game.draw:
            print("Draw !")
            tk.messagebox.showinfo('Game ended', 'Draw !')
            return 0
        elif game.k % 2 == 0 :
            print("Red won !" )
            tk.messagebox.showinfo('Game ended', 'Red won !')
            return 1
        elif game.k % 2 == 1 :
            print("Yellow won !" )
            tk.messagebox.showinfo('Game ended', 'Yellow won !')
            return -1


def play_game(game, player1, player2):
    global move

    # time.sleep(0.5)
    move = make_a_move(game, player1)
    game.push(move, (game.k+1) % 2)
    can.delete('all')
    display_board(game, can)
    can.update()
    move = ()

    if game.game_over:
        return(endgame(game))

    time.sleep(0.5)
    move = make_a_move(game, player2)
    game.push(move, (game.k+1) % 2)
    can.delete('all')
    display_board(game, can)
    can.update()
    move = ()

    if game.game_over:
        return(endgame(game))
        
    can.after(20, play_game, game, player1, player2)


def display_game(player1, player2):
    """
    Starts main function
    """
    global fen, can
    fen = tk.Tk(baseName="Connect4 " + player1.name + " vs " + player2.name)
    can = tk.Canvas(fen, bg='light grey', height=350, width=450) 
    can.pack() 
    game = Connect4() 
    display_board(game, can)
    can.update()
    play_game(game, player1, player2)
    fen.mainloop()
