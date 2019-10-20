import pathlib

import numpy as np
from chess import pgn, Board
from tqdm import tqdm

from utility_module import get_board_state, count_games_from_pgn, uci2onehot

INPUT_FILE_PATH = pathlib.Path("game_data/KingBase2019-A00-A39.pgn")


def get_state_legal_moves(board):
    """Go through all the legal moves of the current board state and return a list of onehot
    vectors"""
    state_legal_moves = []
    for legal_move in board.legal_moves:
        uci = legal_move.uci()
        state_legal_moves.append(uci2onehot(uci=uci))

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
    all_states = []
    for _ in tqdm(range(games_to_get)):
        game = pgn.read_game(pgn_file)
        assert not game.errors

        game_states = get_single_games_states(game=game, return_legal_moves=return_legal_moves)

        if separate_by_game:
            game_states = np.array(game_states)
            all_states.append(game_states)
        else:
            all_states.extend(game_states)

    return all_states


def get_states_from_pgn(input_file, n_games_to_get=None, separate_by_game=True,
                        return_legal_moves=False):
    """Opens specified input pgn file and extracts all states from all games. Returns either a long
    array of all states from the pgn file or a list with an array for each game."""
    # Encoding -> https://python-chess.readthedocs.io/en/latest/pgn.html
    n_games = count_games_from_pgn(input_file=input_file)
    pgn_file = open(input_file, encoding="utf-8-sig")

    if n_games_to_get is None:
        n_games_to_get = n_games

    all_states = get_all_games_states(pgn_file=pgn_file, games_to_get=n_games_to_get,
                                      separate_by_game=separate_by_game,
                                      return_legal_moves=return_legal_moves)

    pgn_file.close()

    if separate_by_game:
        return all_states
    else:
        return np.array(all_states)


get_states_from_pgn(input_file=INPUT_FILE_PATH, n_games_to_get=10, separate_by_game=True,
                    return_legal_moves=True)
