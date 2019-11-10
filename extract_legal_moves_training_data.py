import importlib
import pathlib
from typing import List

import chess
import numpy as np
from chess import pgn, Board
from tqdm import tqdm

import utility_module as ut

importlib.reload(ut)

INPUT_FILE_PATH = pathlib.Path("game_data/KingBase2019-A00-A39.pgn")

# TODO Make sure that the legal moves data is rotated so that it's always from white's perspective
#  I would also like to only have unique states in the legal moves training data

# Todo - I need to find a better way to handle the data as I run out of ram this way. Maybe I should
#  just have a function on top of the neural net that takes a simple representation
#  (like FEN - call board.board_fen() to get), and then convertes to onehot representation.

observed_states = set()


def get_state_legal_moves(board: chess.Board) -> List[np.ndarray]:
    """Go through all the legal moves of the current board state and return a list of onehot
    vectors"""
    state_legal_moves = []
    for legal_move in board.legal_moves:
        uci = legal_move.uci()
        state_legal_moves.append(ut.uci2onehot(uci=uci))

    return state_legal_moves


def get_single_games_states(game, return_legal_moves):
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


def get_all_games_states(pgn_file, games_to_get, separate_by_game, return_legal_moves):
    """Extracts the either a long list of all states from the pgn file or a list for each game with
    a list of states"""
    all_states, all_legal_moves = [], []
    for _ in tqdm(range(games_to_get)):
        game = pgn.read_game(pgn_file)
        assert not game.errors

        game_states = get_single_games_states(game=game, return_legal_moves=return_legal_moves)

        if return_legal_moves:
            game_states, game_legal_moves = game_states
            # game_legal_moves = np.array(game_legal_moves)
            all_legal_moves.append(game_legal_moves)

        if separate_by_game:
            game_states = np.array(game_states)
            all_states.append(game_states)
        else:
            all_states.extend(game_states)

    if return_legal_moves:
        return all_states, all_legal_moves
    else:
        return all_states


def get_states_from_pgn(input_file, n_games_to_get=None, separate_by_game=True,
                        return_legal_moves=False):
    """Opens specified input pgn file and extracts all states from all games. Returns either a long
    array of all states from the pgn file or a list with an array for each game."""
    # Encoding -> https://python-chess.readthedocs.io/en/latest/pgn.html
    n_games = ut.count_games_from_pgn(input_file=input_file)
    pgn_file = open(input_file, encoding="utf-8-sig")

    if n_games_to_get is None:
        n_games_to_get = n_games

    all_legal_moves = None
    all_states = get_all_games_states(pgn_file=pgn_file, games_to_get=n_games_to_get,
                                      separate_by_game=separate_by_game,
                                      return_legal_moves=return_legal_moves)

    if return_legal_moves:
        all_states, all_legal_moves = all_states

    pgn_file.close()

    if not separate_by_game:
        all_states = np.array(all_states)

    if return_legal_moves:
        return all_states, all_legal_moves
    else:
        return all_states


states, legal_moves = get_states_from_pgn(input_file=INPUT_FILE_PATH, n_games_to_get=25,
                                          separate_by_game=True,
                                          return_legal_moves=True)

n_games = len(states)
n_games_val = n_games // 20


def preprocess_legal_move_data(all_states, all_legal_moves):
    out_data_x = []
    out_data_y = []
    for game_states, game_legal_moves in tqdm(list(zip(all_states, all_legal_moves))):

        for state, state_legal_moves in zip(game_states, game_legal_moves):

            for legal_move in state_legal_moves:
                out_data_x.append(state)
                out_data_y.append(legal_move)

    return np.array(out_data_x), np.array(out_data_y)


assert n_games_val > 0
legal_moves_train = preprocess_legal_move_data(all_states=states[:-n_games_val],
                                               all_legal_moves=legal_moves[:-n_games_val])
legal_moves_val = preprocess_legal_move_data(all_states=states[-n_games_val:],
                                             all_legal_moves=legal_moves[-n_games_val:])

# legal_moves_train is now a tuple with input array shape (None, 12, 8, 8) and output array
# (None, 4032) where None is identical in the two cases and corresponds to total number of states
# times mean number of legal moves for each state
