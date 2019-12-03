import importlib
import os
from os import PathLike
from pathlib import Path
from typing import List, Callable, Dict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import utility_module as ut
from resnet_module import ResNet, BasicBlock

importlib.reload(ut)


class NetTrainer:
    """
    Class to handle the training of our neural nets
    """

    def __init__(self, num_classes: int, log_path: Path, loss_function=nn.BCEWithLogitsLoss
                 ) -> None:
        self.num_classes = num_classes
        self.log_path = log_path

        self.net: ResNet = self.initialize_model()
        self.criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = loss_function()
        self.time_step: int = 0
        self.time_steps: List[int] = []
        self.losses: List[float] = []
        self.val_losses: Dict[int, float] = {}
        self.create_log_file()

    def initialize_model(self) -> ResNet:
        model: ResNet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                               num_classes=self.num_classes)
        model.float()
        model.train()
        model.cpu()

        return model

    def evaluate(self, x: torch.Tensor, y: torch.Tensor, is_train_set=True) -> None:
        """Expects to be run within 'with torch.no_grad()'"""
        self.net.eval()

        loss = self.criterion(self.net(x), y)

        if is_train_set:
            self.losses.append(loss.item())
            print(f"train loss: {loss:.3f}")
            self.time_steps.append(self.time_step)
        else:
            self.val_losses[self.time_step] = loss.item()
            print(f"val loss: {loss:.3f}")

        self.net.train()

    def run_epoch(self, x: torch.Tensor, y: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor,
                  optimizer, batch_size: int, evaluate_train_every: int, evaluate_val_every: int
                  ) -> None:
        start_idxs = range(0, x.shape[0], batch_size)
        for minibatch_i, start_idx in tqdm(list(enumerate(start_idxs))):

            optimizer.zero_grad()

            slicer = slice(start_idx, start_idx + batch_size)

            output = self.net(x[slicer])

            loss = self.criterion(output, y[slicer])
            loss.backward()

            with torch.no_grad():
                optimizer.step()

                if not minibatch_i % evaluate_train_every and minibatch_i != 0:
                    self.evaluate(x=x, y=y)

                if evaluate_val_every:
                    if not minibatch_i % evaluate_val_every and minibatch_i != 0:
                        self.evaluate(x=x_val, y=y_val, is_train_set=False)

            self.time_step += 1

    def fit(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, n_epochs: int, lr: float,
            x_val: torch.Tensor, y_val: torch.Tensor, optimizer=torch.optim.Adam,
            evaluate_val_each_epoch: bool = True, evaluate_train_each_epoch: bool = True,
            evaluate_train_every: int = 10, evaluate_val_every: int = 0
            ) -> None:

        optimizer = optimizer(self.net.parameters(), lr=lr)

        self.time_step = 0
        self.losses = []
        self.val_losses = {}
        self.time_steps = []

        for _ in range(n_epochs):
            self.run_epoch(x, y, x_val, y_val, batch_size=batch_size, optimizer=optimizer,
                           evaluate_train_every=evaluate_train_every,
                           evaluate_val_every=evaluate_val_every)

            if evaluate_train_each_epoch:
                self.evaluate(x=x, y=y)
            if evaluate_val_each_epoch:
                self.evaluate(x=x_val, y=y_val, is_train_set=False)

            self.update_log_file()

    def create_log_file(self) -> None:
        assert not self.log_path.exists(), f"Log '{self.log_path.name}' already exists!"
        self.log_path.touch()
        self.log_path.write_text("time_step,loss,type\n")

    def append_to_log_file(self, time_steps: List[int], losses: List[float], loss_type: str
                           ) -> None:
        output_str = ("\n".join([f"{time_step},{loss},{loss_type}"
                                 for time_step, loss in zip(time_steps, losses)]))

        with open(str(self.log_path), 'a') as log_file:
            log_file.write(output_str + "\n")

    def update_log_file(self) -> None:
        """
        Appends latest training results to log file and clears loss variables.
        """
        time_steps_recent_train = self.time_steps[-len(self.losses):]
        time_steps_recent_val, val_losses = zip(*self.val_losses.items())

        self.append_to_log_file(
            time_steps=time_steps_recent_train, losses=self.losses, loss_type='train')
        self.append_to_log_file(
            time_steps=time_steps_recent_val, losses=val_losses, loss_type='val')

        # Clear loss containers
        self.losses = []
        self.val_losses = {}

    def export_training_results(self, train_save_path: PathLike, val_save_path: PathLike) -> None:
        """
        Saves csv files with the training metrics
        """
        df_train_loss = pd.DataFrame(np.c_[self.time_steps, self.losses],
                                     columns=['time_step', 'train_loss'])
        df_val_loss = pd.DataFrame(self.val_losses.items())

        df_train_loss.to_csv(train_save_path), df_val_loss.to_csv(val_save_path)


dp_slicer = slice(100_000)  # N data points to use
N_EPOCHS = 2
LR = 1e-3
BATCH_SIZE = 128

# Load a teporarily saved sample data
X = torch.tensor(joblib.load("./tmp_batch_x")).float()[dp_slicer]
Y = torch.tensor(joblib.load("./tmp_batch_y"))[dp_slicer].float()
fens = joblib.load("./tmp_batch_fens")[dp_slicer]

X_val = torch.tensor(joblib.load("./tmp_val_x")).float()
Y_val = torch.tensor(joblib.load("./tmp_val_y")).float()
val_fens = joblib.load("./tmp_val_fens")

print(X.shape[0], X_val.shape[0])
os.remove('test_test.log')
trainer = NetTrainer(num_classes=4032, log_path=Path('test_test.log'))
trainer.fit(x=X, y=Y, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR, x_val=X_val, y_val=Y_val)

#%%

net = trainer.net
net.eval()
for p in net.parameters():
    p.requires_grad_(False)

# with torch.no_grad():
preds: torch.Tensor = trainer.net(X).sigmoid()
preds_val: torch.Tensor = trainer.net(X_val).sigmoid()

legal_moves = ut.get_nonzero_dict(Y)
legal_moves_val = ut.get_nonzero_dict(Y_val)

ut.get_guessing_score(predictions=preds, legal_moves=legal_moves)
ut.get_guessing_score(predictions=preds_val, legal_moves=legal_moves_val)

# todo
#  - integrate 'get_guessing_score' into code
#  - Consider using fastai library
#  - Set it up so that it can train on (almost?) all of the available games
#  - Make plotting module
