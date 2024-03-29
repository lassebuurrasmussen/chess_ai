import importlib
import pathlib
import random
from typing import List, Tuple

import joblib
import numpy as np
from tqdm import tqdm

import utility_module as ut
from extract_legal_moves import process_pgn, LegalMovesT, FensT

importlib.reload(ut)

INPUT_FILE_PATH = pathlib.Path("game_data/KingBase2019-A00-A39.pgn")

states, legal_moves, fens = process_pgn(input_file=INPUT_FILE_PATH, n_games_to_get=200)


#%%


def preprocess_legal_move_data(games_states: List[np.ndarray], games_legal_moves: LegalMovesT,
                               games_fens: FensT, n_moves: int = 4032
                               ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extracts tuples with arrays of training points. Y is multilabel"""
    out_data_x = []
    out_data_y = []
    out_data_fen = []
    for game_states, game_legal_moves, game_fens in tqdm(list(zip(games_states, games_legal_moves,
                                                                  games_fens))):

        for state, state_legal_moves, state_fen in zip(game_states, game_legal_moves, game_fens):
            # Convert y to multihot vector
            multihot_y = np.bincount(state_legal_moves, minlength=n_moves)

            out_data_x.append(state)
            out_data_y.append(multihot_y)
            out_data_fen.append(state_fen)

    return np.array(out_data_x), np.array(out_data_y), out_data_fen


def train_val_split(all_states: List[np.ndarray], all_legal_moves: LegalMovesT, all_fens,
                    frac_val_games: float = 0.05
                    ) -> Tuple[List[int], List[np.ndarray], LegalMovesT, List[np.ndarray],
                               LegalMovesT, List[List[str]], List[List[str]]]:
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

    train_fens = [all_fens[i] for i in idx_train_games]
    val_fens = [all_fens[i] for i in range(len(all_states)) if i not in idx_train_games]

    return (idx_train_games, train_states, train_legal_moves, val_states, val_legal_moves,
            train_fens, val_fens)


def shuffle_batch(batch_x, batch_y, batch_fens):
    batch_x_idx_shuffled = list(range(len(batch_x)))
    random.shuffle(batch_x_idx_shuffled)
    batch_x = batch_x[batch_x_idx_shuffled]
    batch_y = batch_y[batch_x_idx_shuffled]
    batch_fens = [batch_fens[i] for i in batch_x_idx_shuffled]

    return batch_x, batch_y, batch_fens


def fit_batches(all_states: List[np.ndarray], all_legal_moves: LegalMovesT,
                all_fens: FensT, batch_size: int, frac_val_games: float = 0.05,
                make_sample_pickle: bool = True):
    (idx_train_games, train_states, train_legal_moves, val_states, val_legal_moves,
     train_fens, val_fens) = train_val_split(all_states=all_states, all_legal_moves=all_legal_moves,
                                             all_fens=all_fens, frac_val_games=frac_val_games)

    val_x, val_y, val_fens = preprocess_legal_move_data(games_states=val_states,
                                                        games_legal_moves=val_legal_moves,
                                                        games_fens=val_fens)

    if make_sample_pickle:
        [joblib.dump(var, varname) for var, varname in
         [(val_x, "./tmp_val_x"), (val_y, "./tmp_val_y"), (val_fens, "./tmp_val_fens")]]

    idx_train_games_shuffled = random.sample(range(len(train_states)), len(train_states))
    for batch_i in range(0, len(idx_train_games), batch_size):
        batch_idxs = idx_train_games_shuffled[batch_i:batch_i + batch_size]

        batch_states = [train_states[i] for i in batch_idxs]
        batch_legal_moves = [train_legal_moves[i] for i in batch_idxs]
        batch_fens = [train_fens[i] for i in batch_idxs]

        batch_x, batch_y, batch_fens = preprocess_legal_move_data(
            games_states=batch_states, games_legal_moves=batch_legal_moves, games_fens=batch_fens)

        # Shuffle data
        batch_x, batch_y, batch_fens = shuffle_batch(batch_x=batch_x, batch_y=batch_y,
                                                     batch_fens=batch_fens)

        if batch_i == 0 and make_sample_pickle:
            [joblib.dump(var, varname) for var, varname in [(batch_x, "./tmp_batch_x"),
                                                            (batch_y, "./tmp_batch_y"),
                                                            (batch_fens, "./tmp_batch_fens")]]
            raise Exception


fit_batches(all_states=states, all_legal_moves=legal_moves, all_fens=fens, batch_size=200)

# Todo:
#  Implement Net.
