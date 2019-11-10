import importlib
import pathlib
import random
from os import PathLike
from typing import List, Optional, Union, Tuple, TextIO

import chess
import numpy as np
from chess import pgn, Board
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

import utility_module as ut
from neural_nets import Net
from utility_module import ALL_MOVES_1D

importlib.reload(ut)

INPUT_FILE_PATH = pathlib.Path("game_data/KingBase2019-A00-A39.pgn")

# TODO Verify that code works before I start training


observed_states = set()


def get_state_legal_moves(board: chess.Board) -> List[int]:
    """Go through all the legal moves of the current board state and return a list of onehot
    vectors"""
    state_legal_moves = []
    for legal_move in board.legal_moves:
        uci = legal_move.uci()
        state_legal_moves.append(ut.uci2onehot(uci=uci))

    return state_legal_moves


def add_board_state_to_list(board: chess.Board, in_list: list) -> None:
    """Extracts state from board and appends to input list"""
    state = ut.get_board_state(in_board=board)
    in_list.append(state)


def add_if_known(board: chess.Board, game_legal_moves: List[List[int]],
                 game_states: List[np.ndarray]) -> None:
    """If board state hasn't already been observed:
    Adds state to set of observed states. Adds legal moves corresponding to the Board."""
    board_fen = board.board_fen()
    if board_fen not in observed_states:
        observed_states.add(board_fen)

        state_legal_moves = get_state_legal_moves(board=board)
        game_legal_moves.append(state_legal_moves)

        add_board_state_to_list(board=board, in_list=game_states)


def get_single_games_states(game: chess.pgn.Game, return_legal_moves: bool):
    """Create new chess.Board instance and plays game till the end. Returns list of array of all
    states along the way.
    Can also return list of legal moves per state"""
    board = Board()
    game_states = []
    game_legal_moves = []
    white_turn = True
    for move in game.mainline_moves():

        if return_legal_moves:
            # Only add board position to data if it hasn't been observed
            board_to_save = board if white_turn else board.mirror()

            add_if_known(board=board_to_save, game_legal_moves=game_legal_moves,
                         game_states=game_states)
            white_turn = not white_turn

        else:
            # Add board position to data irrespective of whether it's been observed
            add_board_state_to_list(board=board, in_list=game_states)

        board.push(move)

    if return_legal_moves:
        return game_states, game_legal_moves

    else:
        # Get last state
        # (As it's only relevant when not looking at legal moves as there are no more moves)
        add_board_state_to_list(board=board, in_list=game_states)

        return game_states


def get_all_games_states(pgn_file: TextIO, games_to_get: int, return_legal_moves: bool,
                         show_progress: bool
                         ) -> Union[Tuple[List[np.ndarray],
                                          List[List[int]]],
                                    List[np.ndarray]]:
    """Extracts a list for each game with a list of states"""
    all_states, all_legal_moves = [], []
    iterator = tqdm(range(games_to_get)) if show_progress else range(games_to_get)
    for _ in iterator:
        game = pgn.read_game(pgn_file)
        assert not game.errors

        game_states = get_single_games_states(game=game, return_legal_moves=return_legal_moves)

        if return_legal_moves:
            game_states, game_legal_moves = game_states
            # game_legal_moves = np.array(game_legal_moves)
            all_legal_moves.append(game_legal_moves)

        game_states = np.array(game_states)
        all_states.append(game_states)

    if return_legal_moves:
        return all_states, all_legal_moves
    else:
        return all_states


def get_states_from_pgn(input_file: PathLike, n_games_to_get: Optional[int] = None,
                        return_legal_moves: bool = False, show_progress: bool = False
                        ) -> Union[Tuple[List[np.ndarray], List[List[str]]],
                                   List[np.ndarray]]:
    """Opens specified input pgn file and extracts all states from all games.."""
    # Encoding used comes from https://python-chess.readthedocs.io/en/latest/pgn.html
    n_games = ut.count_games_from_pgn(input_file=input_file)
    pgn_file = open(input_file, encoding="utf-8-sig")

    if n_games_to_get is None:
        n_games_to_get = n_games

    all_legal_moves = None
    all_states = get_all_games_states(pgn_file=pgn_file, games_to_get=n_games_to_get,
                                      return_legal_moves=return_legal_moves,
                                      show_progress=show_progress)

    if return_legal_moves:
        # Unpack returned variable
        all_states, all_legal_moves = all_states

    pgn_file.close()

    if return_legal_moves:
        return all_states, all_legal_moves
    else:
        return all_states


states, legal_moves = get_states_from_pgn(input_file=INPUT_FILE_PATH, n_games_to_get=20,
                                          return_legal_moves=True)

assert len(states) == len(legal_moves)
assert all([len(st) == len(leg) for st, leg in zip(states, legal_moves)])


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
