import importlib
import pathlib
from os import PathLike
from typing import List, Optional, Union, Tuple, TextIO, Set

import chess
import numpy as np
from chess import pgn, Board
from tqdm import tqdm

import utility_module as ut

importlib.reload(ut)

INPUT_FILE_PATH = pathlib.Path("game_data/KingBase2019-A00-A39.pgn")

# Type defining
LegalMovesT = List[List[int]]
FensT = List[List[str]]

observed_states_set: Set[str] = set()
observed_states = []


def get_state_legal_moves(board: chess.Board) -> List[int]:
    """Go through all the legal moves of the current board state and return a list of onehot
    vector indices"""
    state_legal_moves = []
    for legal_move in board.legal_moves:
        uci = legal_move.uci()
        state_legal_moves.append(ut.uci2onehot_idx(uci=uci))

    return state_legal_moves


def add_board_state_to_list(board: chess.Board, in_list: list) -> None:
    """Extracts state from board and appends to input list"""
    state = ut.get_board_state(in_board=board)
    in_list.append(state)


def add_if_known(board: chess.Board, game_legal_moves: LegalMovesT,
                 game_states: List[np.ndarray]) -> None:
    """If board state hasn't already been observed: Adds state to set and list of observed states.
    Also adds legal moves and state corresponding to the Board."""
    board_fen = board.fen()
    if board_fen not in observed_states_set:
        observed_states_set.add(board_fen)
        observed_states.append(board_fen)

        state_legal_moves = get_state_legal_moves(board=board)
        game_legal_moves.append(state_legal_moves)

        add_board_state_to_list(board=board, in_list=game_states)


def get_single_games_states(game: chess.pgn.Game, return_legal_moves: bool
                            ) -> Union[Tuple[List[np.ndarray], LegalMovesT],
                                       List[np.ndarray]]:
    """Create new chess.Board instance and plays game till the end. Returns list of array of all
    states along the way.
    Can also return list of legal moves per state"""
    board = Board()
    game_states: List[np.ndarray] = []
    game_legal_moves: LegalMovesT = []
    white_turn = True  # Keep track of who's turn it is

    for move_i, move in enumerate(game.mainline_moves()):

        if return_legal_moves:
            board_to_save = board if white_turn else board.mirror()

            # Only add board position to data if it hasn't been observed
            add_if_known(board=board_to_save, game_legal_moves=game_legal_moves,
                         game_states=game_states)

            # Next player's turn
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
                                          LegalMovesT],
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
            all_legal_moves.append(game_legal_moves)

        game_states = np.array(game_states)
        all_states.append(game_states)

    if return_legal_moves:
        return all_states, all_legal_moves
    else:
        return all_states


def get_states_from_pgn(input_file: PathLike, n_games_to_get: Optional[int] = None,
                        return_legal_moves: bool = False, show_progress: bool = False
                        ) -> Union[Tuple[List[np.ndarray], LegalMovesT],
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


def process_pgn(input_file: PathLike, n_games_to_get: int
                ) -> Tuple[List[np.ndarray], LegalMovesT, FensT]:
    # Keep both a set and a list to save time on "in" operation, but preserve order.
    # Obviously at the cost of more memory

    global observed_states_set
    global observed_states

    all_states, all_legal_moves = get_states_from_pgn(input_file=input_file,
                                                      n_games_to_get=n_games_to_get,
                                                      return_legal_moves=True)

    assert len(all_states) == len(all_legal_moves)
    assert [len(s) for s in all_states] == [len(leg) for leg in all_legal_moves]
    assert len(observed_states) == sum([len(s) for s in all_states])

    # Split observed states by game
    all_fens = ut.split_list_as(in_list=observed_states, template_list=all_states)
    assert [len(obs) for obs in all_fens] == [len(s) for s in all_states]

    return all_states, all_legal_moves, all_fens
