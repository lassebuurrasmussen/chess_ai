import re
from itertools import permutations
from os import PathLike
from typing import Tuple, List, Optional, Dict

import chess
import numpy as np
import torch
from chess import SQUARES_180

# noinspection PyUnresolvedReferences
from useful_objects import ONEHOT_TEMPLATE_ARRAY

PIECES_WHITE = ['P', 'R', 'N', 'B', 'Q', 'K']

PIECES_BLACK = list("".join(PIECES_WHITE).lower())

PIECE_INT_LOOKUP = {p: i for i, p in enumerate(PIECES_WHITE + PIECES_BLACK)}

POSITION_LOOKUP = [[0, 1, 2, 3, 4, 5, 6, 7],
                   [8, 9, 10, 11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20, 21, 22, 23],
                   [24, 25, 26, 27, 28, 29, 30, 31],
                   [32, 33, 34, 35, 36, 37, 38, 39],
                   [40, 41, 42, 43, 44, 45, 46, 47],
                   [48, 49, 50, 51, 52, 53, 54, 55],
                   [56, 57, 58, 59, 60, 61, 62, 63]]

UCI_LETTER_LOOKUP = {97: '0', 98: '1', 99: '2', 100: '3', 101: '4', 102: '5', 103: '6', 104: '7'}

ALL_MOVES_1D = list(permutations(range(8 * 8), r=2))


def count_games_from_pgn(input_file: PathLike) -> int:
    """Reads in PGN game file and counts the number of occurrences of 'EventDate'"""
    with open(input_file, encoding="latin") as f:
        file = f.read()
    return len(re.findall("EventDate", file))


def get_board_state(in_board: chess.Board) -> np.ndarray:
    """Function adapted from chess.Board.__str__() to generate a numpy array representing the board
    state"""
    # 6 pieces * 2 colors + 2 for en passant and castling. 8*8 fields of the board.
    builder = np.zeros([12 + 2, 8 * 8])
    for square in SQUARES_180:

        piece = in_board.piece_at(square)

        if piece:
            piece_int = PIECE_INT_LOOKUP[piece.symbol()]
            builder[piece_int, square] = 1

    builder = builder.reshape([14, 8, 8])

    # As the SQUARES_180 is flipped I need to flip it back
    builder = np.flip(builder, 1)

    if in_board.has_legal_en_passant():
        en_passant_coords = get_en_passant_coords(fen=in_board.fen())
        builder[12, en_passant_coords[0], en_passant_coords[1]] = 1

    if in_board.has_castling_rights(color=True) or in_board.has_castling_rights(color=False):
        castling_ints = get_castling_ints(castling_fen=in_board.castling_shredder_fen())
        builder[13, 0, castling_ints] = 1

    return builder


def white_to_black(in_state: np.ndarray, single_state: bool = False) -> np.ndarray:
    """Takes an input state array and swaps the 6 first matrices with the 6 last matrices. This has
    the effect of seeing the white pieces as black and vice versa."""
    in_state = in_state.copy()

    n_pieces = len(in_state) if single_state else len(in_state[0])
    n_white_pieces = n_pieces // 2

    normal_order = list(range(n_pieces))
    new_order = list(range(n_white_pieces, n_pieces)) + list(range(n_white_pieces))

    if single_state:
        in_state[normal_order] = in_state[new_order]
    else:
        in_state[:, normal_order] = in_state[:, new_order]

    return in_state


def mirror_state(in_state: np.ndarray, single_state=False) -> np.ndarray:
    """Takes a state from perspective of white and outputs perspective black and vice versa"""
    in_state = white_to_black(in_state, single_state=single_state)

    if single_state:
        return np.flip(in_state, axis=1)
    else:
        return np.flip(in_state, axis=2)


def coordinates_to_onehot_index(entry: Tuple[int, int]) -> int:
    """Don't ask me why this works. Makes my head explode. But it's much faster than calling .index
    method on list.

    Can be tested with:
    assert all([coordinates_to_onehot_index(entry) == i for i, entry in enumerate(one_hot)])"""
    return (entry[0]) * 63 + entry[1] - 1 + int(entry[0] > entry[1])


def uci_translate(uci: str):
    """
    Translate UCI move to integers e.g. 'g' is has unicode 103 and is the index 6 column on the
    # board. So g -> 6
    """
    return uci.translate(UCI_LETTER_LOOKUP)


def uci2onehot_idx(uci: str) -> int:
    """Takes UCI move str and converts it to index of a onehot vector of all moves"""

    move_translated = uci_translate(uci=uci)

    # Extract source and destination coordinates
    # swap axes as UCI is column first instead of row first
    src, dest = ((move_translated[1], move_translated[0]),
                 (move_translated[3], move_translated[2]))

    # Convert to ints
    src, dest = [int(src[0]), int(src[1])], [int(dest[0]), int(dest[1])]

    # Subtract each row index from 8 as UCI counts last row as first
    src[0] = 8 - src[0]
    dest[0] = 8 - dest[0]

    # Convert 2D coordinates to 1D coordinates
    src_int = POSITION_LOOKUP[src[0]][src[1]]
    dest_int = POSITION_LOOKUP[dest[0]][dest[1]]

    # Find index of source and destination in 1d move vector
    hot = coordinates_to_onehot_index((src_int, dest_int))

    # # Uncomment to instead convert directly to one-hot
    # onehot = ONEHOT_TEMPLATE_ARRAY.copy()
    # onehot[hot] = 1

    return hot


def split_list_as(in_list: List, template_list: List[List]) -> List[List]:
    """Takes a long list and subdivides it as the provided template list"""
    splitted_list = []
    total_length = 0
    for game_length in [len(s) for s in template_list]:
        splitted_list.append(in_list[total_length:total_length + game_length])
        total_length += game_length

    return splitted_list


def get_en_passant_coords(fen: str) -> Optional[List[int]]:
    """
    Returns the coordinates of the en passant square if it exists
    """
    en_passant_field = fen.split()[3]

    if en_passant_field == "-":
        # fen contains "-" in fourth position and thereby no en passant
        return None

    en_passant_coords = uci_translate(uci=en_passant_field)

    # Swap axes as UCI is column first instead of row first
    en_passant_coords = en_passant_coords[::-1]

    # Convert to ints
    en_passant_coords = [int(en_passant_coords[0]), int(en_passant_coords[1])]

    # Subtract each row index from 8 as UCI counts last row as first
    en_passant_coords[0] = 8 - en_passant_coords[0]

    return en_passant_coords


def get_castling_ints(castling_fen: str) -> List[int]:
    """
    Returns integers corresponding to the 4 different castling rights
    """

    translater = {ord(l): ord(i) for l, i in zip('HAha', '0123')}
    return [int(i) for i in castling_fen.translate(translater)]


def get_guessing_score(predictions: torch.Tensor, legal_moves: Dict[int, List[int]]) -> float:
    """
    Calculates a score in range 0-1 based on how many of the top guesses are needed to guess all the
    legal moves of a state
    """
    # Argmax of predictions
    predictions_sorted = predictions.sort(dim=1, descending=True)[1]

    guesses, n_legal_moves_all = [], []
    for row, board_legal_moves in legal_moves.items():
        row_prediction = predictions_sorted[row]
        correct_predictions = np.isin(row_prediction, board_legal_moves)

        n_legal_moves = len(board_legal_moves)

        # Count the number of guesses it takes to guess all the legal moves
        cumulative_sum = correct_predictions.cumsum()
        all_legal_moves_guessed = np.where(cumulative_sum == n_legal_moves)[0]

        # Take the first integer where all the moves have been guessed and append
        guesses.append(all_legal_moves_guessed[0])

        # Save the number of legal moves for each row
        n_legal_moves_all.append(n_legal_moves)

    # Calculate the ratio between number of legal moves and guesses it took
    scores = np.array(n_legal_moves_all) / (np.array(guesses) + 1)  # (+1 because of zero-indexing)

    return scores.mean()


def get_nonzero_dict(tensor: torch.Tensor) -> Dict[int, List[int]]:
    """
    Converts a tensor to a dictionary with lists of all indices of entries that are non-zero
    """
    d = {}
    for row, val in tensor.nonzero(as_tuple=False):
        row, val = row.item(), val.item()
        if row not in d.keys():
            d[row] = [val]
        else:
            d[row].append(val)

    return d
