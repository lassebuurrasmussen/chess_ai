import pathlib

import joblib
import numpy as np
from scipy import sparse

from extract_legal_moves_training_data import get_states_from_pgn
from utility_module import mirror_state

INPUT_FILE_PATH = pathlib.Path("game_data/KingBase2019-A00-A39.pgn")

states = get_states_from_pgn(INPUT_FILE_PATH, n_games_to_get=100, separate_by_game=True)

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
