import re
from itertools import permutations

import chess
import numpy as np
from chess import SQUARES_180

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


def count_games_from_pgn(input_file):
    """Reads in PGN game file and counts the number of occurrences of 'EventDate'"""
    with open(input_file, encoding="latin") as f:
        file = f.read()
    return len(re.findall("EventDate", file))


def get_board_state(in_board: chess.Board):
    """Function adapted from chess.Board.__str__() to generate a numpy array representing the board
    state"""
    builder = np.zeros([12, 8 * 8])
    for square in SQUARES_180:

        piece = in_board.piece_at(square)

        if piece:
            piece_int = PIECE_INT_LOOKUP[piece.symbol()]
            builder[piece_int, square] = 1

    builder = builder.reshape([12, 8, 8])

    # As the SQUARES_180 is flipped I need to flip it back
    return np.flip(builder, 1)


def white_to_black(a, single_state=False):
    """Takes an input state array and swaps the 6 first matrices with the 6 last matrices. This has
    the effect of seeing the white pieces as black and vice versa."""
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
    """Takes a state from perspective of white and outputs perspective black and vice versa"""
    a = white_to_black(a, single_state=single_state)

    if single_state:
        return np.flip(a, axis=1)
    else:
        return np.flip(a, axis=2)


def coordinates_to_onehot_index(entry):
    """Don't ask me why this works. Makes my head explode. But it's much faster than calling .index
    method on list.

    Can be tested with:
    assert all([coordinates_to_onehot_index(entry) == i for i, entry in enumerate(one_hot)])"""
    # return (entry[0] - 1) * 63 + entry[1] - 2 + int(entry[0] > entry[1])
    return (entry[0]) * 63 + entry[1] - 1 + int(entry[0] > entry[1])


def uci2onehot(uci: str):
    """Takes UCI move str and converts it to a onehot vector"""
    # Translate UCI move to integers
    move_translated = uci.translate(UCI_LETTER_LOOKUP)

    # Extract source and destination coordinates
    src, dest = ((move_translated[1], move_translated[0]),
                 (move_translated[3], move_translated[2]))
    src, dest = [int(src[0]), int(src[1])], [int(dest[0]), int(dest[1])]

    # Subtract each row index from 8 as UCI counts last row as first
    src[0] = 8 - src[0]
    dest[0] = 8 - dest[0]

    # Convert 2D coordinates to 1D coordinates
    src_int = POSITION_LOOKUP[src[0]][src[1]]
    dest_int = POSITION_LOOKUP[dest[0]][dest[1]]

    # Find index of source and destination in 1d move vector
    hot = coordinates_to_onehot_index((src_int, dest_int))
    # onehot = ONEHOT_TEMPLATE.copy()
    onehot = ONEHOT_TEMPLATE_ARRAY.copy()

    onehot[hot] = 1

    return onehot
