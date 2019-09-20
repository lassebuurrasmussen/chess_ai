# Chess AI

I am going to treat this README as a journal for now too keep track of where I am and what I have to do.

The first step is to get a chess engine so we can play chess!
I Decided to go with [Sunfish](https://github.com/thomasahle/sunfish) (literally the first one I found).

My idea for now is to use off-policy q-learning and see where it takes me.
That requires a behacvioural policy that we can observe.
For that I will just use the in-built policy to begin with.
This means I'm going to have the game play itself.

I've modified the code so that the MTD-bi search algorithm now plays against itself.
However, the issue is now that it gets stuck in loops.

I've found that there are more games online that I could ever need! I downladed 2.2 million games from https://www.kingbase-chess.net/. They are of the PGN format, which is not the same as what my chess engine takes. So I'm using python-chess to convert the moves to the correct format. I can convert around 250-290 games per second. This is much more than I could simulate per second if the games have to be of decent quality! I've converted 10,000 games and dumped them [here](./game_data/KingBase2019-A00-A39_10000_games.dump). 