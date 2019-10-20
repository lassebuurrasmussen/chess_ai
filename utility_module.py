import chess
import numpy as np
import re

from chess import SQUARES_180

PIECES_WHITE = ['P', 'R', 'N', 'B', 'Q', 'K']
PIECES_BLACK = list("".join(PIECES_WHITE).lower())
PIECE_INT_LOOKUP = {p: i for i, p in enumerate(PIECES_WHITE + PIECES_BLACK)}


def count_games_from_pgn(input_file):
    """Reads in PGN game file and counts the number of occurrences of 'EventDate'"""
    with open(input_file, encoding="latin") as f:
        file = f.read()
    return len(re.findall("EventDate", file))


def board_2_array(in_board: chess.Board):
    """Function adapted from chess.Board.__str__() to generate a numpy array representing the board
    state"""
    builder = np.zeros([12, 8 * 8])
    for square in SQUARES_180:

        piece = in_board.piece_at(square)

        if piece:
            piece_int = PIECE_INT_LOOKUP[piece.symbol()]
            builder[piece_int, square] = 1

    builder = builder.reshape([12, 8, 8])
    return np.flip(builder, 1)  # Not sure why, but I need to flip for it to match


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
