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
