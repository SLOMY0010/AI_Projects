"""
Tic Tac Toe Player
"""

import copy
import math
import random


X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    xcount = ocount = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                xcount += 1
            elif board[i][j] == O:
                ocount += 1

    return O if xcount > ocount else X
    


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    freeSquares = set()
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                freeSquares.add((i, j))
    return freeSquares


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # if board[action[0]][action[1]] != EMPTY:
    #     raise Exception("illegal move.")
    
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    tem_board = copy.deepcopy(board)
    #converting every X to True and every O to False for easier representation
    for i in range(3):
        for j in range(3):
            if tem_board[i][j] == X:
                tem_board[i][j] = True
            elif tem_board[i][j] == O:
                tem_board[i][j] = False

    #winning positions
    diagonals = ([tem_board[0][0], tem_board[1][1], tem_board[2][2]], [tem_board[0][2], tem_board[1][1], tem_board[2][0]])
    rows = (tem_board[0], tem_board[1], tem_board[2])
    columns = ([rows[0][0], rows[1][0], rows[2][0]], [rows[0][1], rows[1][1], rows[2][1]], [rows[0][2], rows[1][2], rows[2][2]])

    #checking for a win in the diagonals
    for i in range(len(diagonals)):
        if all(diagonals[i]):
            return X
        elif not any(diagonals[i]) and EMPTY not in diagonals[i]:
            return O

    #checking for a win in the rows
    for i in range(len(rows)):
        if all(rows[i]):
            return X
        elif not any(rows[i]) and EMPTY not in rows[i]:
            return O

    #checking for a win in the columns
    for i in range(len(columns)):
        if all(columns[i]):
            return X
        elif not any(columns[i]) and EMPTY not in columns[i]:
            return O

    #if there is no win return none
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    #return True if there is a winner
    if winner(board) == X or winner(board) == O:
        return True

    #return False if there is any available square
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                return False

    #return Trueif its a tie
    return True




def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    moves = []
    evals_for_moves = []

    if terminal(board):
        return None

    #storing every legal move in a list
    for action in actions(board):
        moves.append(action)

    '''
    if the AI is X, and it is the first move return
    any random move, because all moves have the same
    evaluation of 0, and for the sake of saving time.
    '''
    if player(board) == X and len(moves) == 9:
        return moves[random.randrange(0, 8)]

    #storing the evaluation of every possible move in a list based on the player.
    for action in actions(board):

        if player(board) == X:
            current_eval = max_value(result(board, action))
            evals_for_moves.append(current_eval)
            if current_eval > evals_for_moves[-1]:
                return action

        elif player(board) == O:
            current_eval = min_value(result(board, action))
            evals_for_moves.append(current_eval)
            if current_eval < evals_for_moves[-1]:
                return action

    #return the best move based on the player
    if player(board) == X:
        for i in range(len(evals_for_moves)):
            if evals_for_moves[i] == max(evals_for_moves):
                return moves[i]

    elif player(board) == O:
        for i in range(len(evals_for_moves)):
            if evals_for_moves[i] == min(evals_for_moves):
                return moves[i]


def max_value(board):
    if terminal(board):
        return utility(board)
    v = float('inf')
    for action in actions(board):
        v = min(v, min_value(result(board, action)))

        #this statement is to implement Alpha-Beta pruning.
        if v == -1:
            return v

    return v


def min_value(board):
    if terminal(board):
        return utility(board)
    v = float('-inf')
    for action in actions(board):
        v = max(v, max_value(result(board, action)))

        #this statement is to implement Alpha-Beta pruning.
        if v == 1:
            return v

    return v