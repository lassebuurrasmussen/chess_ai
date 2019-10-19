import numpy as np
from chess import pgn, Board
from tqdm import tqdm

from utility_module import board_2_array, count_games_from_pgn

GAMES_TO_READ = 10_000
INPUT_FILE = "./game_data/KingBase2019-A00-A39.pgn"


def get_states_from_pgn(input_file, games_to_get=None):
    # Encoding -> https://python-chess.readthedocs.io/en/latest/pgn.html
    n_games = count_games_from_pgn(input_file=input_file)
    pgn_file = open(input_file, encoding="utf-8-sig")

    if games_to_get is None:
        games_to_get = n_games

    states = []
    # for game_i in range(GAMES_TO_READ)[0:10]:
    for game_i in tqdm(range(games_to_get)):

        game = pgn.read_game(pgn_file)
        assert not game.errors

        board = Board()
        for move in game.mainline_moves():
            state = board_2_array(board)
            states.append(state)

            board.push(move)

        state = board_2_array(board)
        states.append(state)
    states = np.array(states)
