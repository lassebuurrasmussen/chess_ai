import numpy as np
from chess import pgn, Board
from tqdm import tqdm
from scipy import sparse
import joblib
import pathlib

from utility_module import board_2_array, count_games_from_pgn

INPUT_FILE_PATH = pathlib.Path("game_data/KingBase2019-A00-A39.pgn")


def white_to_black(a, single_state=False):
    a = a.copy()

    n_pieces = len(a) if single_state else len(a[0])
    n_white_pieces = n_pieces // 2

    normal_order = list(range(n_pieces))
    new_order = list(range(n_white_pieces, n_pieces)) + list(range(n_white_pieces))

    if single_state:
        a[normal_order] = a[new_order]
    else:
        a[:, normal_order] = a[:, new_order]

    return a


def mirror_state(a, single_state=False):
    a = white_to_black(a, single_state=single_state)

    if single_state:
        return np.flip(a, axis=1)
    else:
        return np.flip(a, axis=2)


def get_states_from_pgn(input_file, games_to_get=None):
    # Encoding -> https://python-chess.readthedocs.io/en/latest/pgn.html
    n_games = count_games_from_pgn(input_file=input_file)
    pgn_file = open(input_file, encoding="utf-8-sig")

    if games_to_get is None:
        games_to_get = n_games

    states = []
    for _ in tqdm(range(games_to_get)):

        game = pgn.read_game(pgn_file)
        assert not game.errors

        board = Board()
        for move in game.mainline_moves():
            state = board_2_array(board)
            states.append(state)

            board.push(move)

        state = board_2_array(board)
        states.append(state)

    pgn_file.close()

    return np.array(states)


states = get_states_from_pgn(INPUT_FILE_PATH, games_to_get=100)

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
