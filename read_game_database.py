from chess import pgn
from tqdm import tqdm
import joblib

GAMES_TO_READ = 10_000
INPUT_FILE = "./game_data/KingBase2019-A00-A39.pgn"
OUTPUT_FILE = INPUT_FILE.replace('.pgn', f'_{GAMES_TO_READ}_games.dump')

pgn_file = open(
    INPUT_FILE,
    encoding="utf-8-sig")  # Encoding -> https://python-chess.readthedocs.io/en/latest/pgn.html

games_moves = []
for game_i in tqdm(range(GAMES_TO_READ)):
    game = pgn.read_game(pgn_file)

    try:
        games_moves.append([move.uci() for move in game.mainline_moves()])
    except AttributeError:
        print(f"Done at game {game_i}")
        break

joblib.dump(games_moves, OUTPUT_FILE)
