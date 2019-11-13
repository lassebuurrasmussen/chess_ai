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

observed_states = set()

LegalMovesT = List[List[int]]


def get_state_legal_moves(board: chess.Board) -> List[int]:
    """Go through all the legal moves of the current board state and return a list of onehot
    vectors"""
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
    for move_i, move in enumerate(game.mainline_moves()):

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


#%%

def preprocess_legal_move_data(games_states: List[np.ndarray], games_legal_moves: LegalMovesT,
                               ohe: OneHotEncoder) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts tuples with arrays of training points."""
    out_data_x = []
    out_data_y = []
    for game_states, game_legal_moves in tqdm(list(zip(games_states, games_legal_moves))):

        for state, state_legal_moves in zip(game_states, game_legal_moves):

            for legal_move in state_legal_moves:
                out_data_x.append(state)
                out_data_y.append(legal_move)

    return np.array(out_data_x), (ohe.transform(np.array(out_data_y).reshape(-1, 1)))


def train_val_split(all_states: List[np.ndarray], all_legal_moves: LegalMovesT,
                    frac_val_games: float = 0.05
                    ) -> Tuple[List[int], List[np.ndarray], LegalMovesT, List[np.ndarray],
                               LegalMovesT]:
    """Splits games into training and validation games"""
    random.seed(3947)
    n_games = len(states)
    n_games_val = int(n_games * frac_val_games)
    idx_val_games = random.sample(range(n_games), k=n_games_val)
    idx_train_games = [i for i in range(n_games) if i not in idx_val_games]

    train_states = [all_states[i] for i in range(n_games) if i in idx_train_games]
    train_legal_moves = [all_legal_moves[i] for i in range(n_games) if i in idx_train_games]

    val_states = [all_states[i] for i in range(n_games) if i in idx_val_games]
    val_legal_moves = [all_legal_moves[i] for i in range(n_games) if i in idx_val_games]

    return idx_train_games, train_states, train_legal_moves, val_states, val_legal_moves


def make_onehot_encoder():
    """Remember to use ohe to de-transform afterwards, as it is not certain to correspond to int
    values"""
    n_moves = len(ALL_MOVES_1D)
    return OneHotEncoder(categories='auto', sparse=False).fit(np.arange(n_moves).reshape(-1, 1))


def fit_batches(all_states: List[np.ndarray], all_legal_moves: LegalMovesT, batch_size: int,
                frac_val_games: float = 0.05):
    idx_train_games, train_states, train_legal_moves, val_states, val_legal_moves = train_val_split(
        all_states=all_states, all_legal_moves=all_legal_moves, frac_val_games=frac_val_games)

    ohe = make_onehot_encoder()

    val_x, val_y = preprocess_legal_move_data(games_states=val_states,
                                              games_legal_moves=val_legal_moves,
                                              ohe=ohe)

    idx_train_games_shuffled = random.sample(idx_train_games, len(idx_train_games))
    for batch_i in range(0, len(idx_train_games), batch_size):
        batch_idxs = idx_train_games_shuffled[batch_i:batch_i + batch_size]

        batch_states = [train_states[i] for i in batch_idxs]
        batch_legal_moves = [train_legal_moves[i] for i in batch_idxs]

        batch_x, batch_y = preprocess_legal_move_data(games_states=batch_states,
                                                      games_legal_moves=batch_legal_moves, ohe=ohe)

        model = Net()
        model.fit(x=batch_x, y=batch_y)


fit_batches(all_states=states, all_legal_moves=legal_moves, batch_size=5)

# Todo:
#  Implement Net.
