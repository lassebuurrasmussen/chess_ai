import pathlib

import joblib
import numpy as np
from chess import pgn, Board
from scipy import sparse
from tqdm import tqdm

from utility_module import get_board_state, count_games_from_pgn, mirror_state, uci2onehot

INPUT_FILE_PATH = pathlib.Path("game_data/KingBase2019-A00-A39.pgn")


def get_state_legal_moves(board):
    """Go through all the legal moves of the current board state and return a list of onehot
    vectors"""
    state_legal_moves = []
    for legal_move in board.legal_moves:
        uci = legal_move.uci()
        state_legal_moves.append(uci2onehot(uci=uci))

    return state_legal_moves


def get_single_games_states(game, return_legal_moves=False):
    """Create new chess.Board instance and plays game till the end. Returns list of array of all
    states along the way.
    Can also return list of legal moves per state"""
    board = Board()
    game_states = []
    game_legal_moves = []
    for move in game.mainline_moves():
        state = get_board_state(board)

        if return_legal_moves:
            state_legal_moves = get_state_legal_moves(board=board)
            game_legal_moves.append(state_legal_moves)

        game_states.append(state)

        board.push(move)

    # Get last state
    state = get_board_state(board)
    game_states.append(state)

    if return_legal_moves:
        return game_states, game_legal_moves
    else:
        return game_states


def get_all_games_states(pgn_file, games_to_get, separate_by_game):
    """Extracts the either a long list of all states from the pgn file or a list for each game with
    a list of states"""
    all_states = []
    for _ in tqdm(range(games_to_get)):
        game = pgn.read_game(pgn_file)
        assert not game.errors

        game_states = get_single_games_states(game=game)

        if separate_by_game:
            game_states = np.array(game_states)
            all_states.append(game_states)
        else:
            all_states.extend(game_states)

    return all_states


def get_states_from_pgn(input_file, games_to_get=None, separate_by_game=True):
    """Opens specified input pgn file and extracts all states from all games. Returns either a long
    array of all states from the pgn file or a list with an array for each game."""
    # Encoding -> https://python-chess.readthedocs.io/en/latest/pgn.html
    n_games = count_games_from_pgn(input_file=input_file)
    pgn_file = open(input_file, encoding="utf-8-sig")

    if games_to_get is None:
        games_to_get = n_games

    all_states = get_all_games_states(pgn_file=pgn_file, games_to_get=games_to_get,
                                      separate_by_game=separate_by_game)

    pgn_file.close()

    if separate_by_game:
        return all_states
    else:
        return np.array(all_states)


states = get_states_from_pgn(INPUT_FILE_PATH, games_to_get=100, separate_by_game=True)
#%%


input_ixs = list(range(len(states) - 1))
output_ixs = list(range(1, len(states)))
input_output_pairs = np.stack([input_ixs, output_ixs], axis=1)

data_output = states[input_output_pairs]
data_output_white = data_output[0::2]
data_output_black = data_output[1::2]

data_output_black = mirror_state(data_output_black)


def do_this_later():
    shape = states.shape
    states_sparse = sparse.csr_matrix(states.flatten())

    shape_str = str(shape).strip("'()").translate({ord(','): "", ord(' '): '-'})

    joblib.dump(states_sparse, f"{INPUT_FILE_PATH.stem}_shape{shape_str}.dump")
