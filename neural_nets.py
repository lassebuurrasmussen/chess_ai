from collections import defaultdict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from resnet_module import ResNet, BasicBlock


class NetTrainer:

    def __init__(self, num_classes: int, loss_function=nn.BCEWithLogitsLoss) -> None:
        self.num_classes = num_classes

        self.net: ResNet = self.initialize_model()
        self.criterion = loss_function()
        self.time_step = 0
        self.time_steps = []
        self.losses = []
        self.val_losses = {}

    def initialize_model(self) -> ResNet:
        model: nn.Module = ResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                                  num_classes=self.num_classes)
        model.float()
        model.train()
        model.cpu()

        return model

    def evaluate(self, x: torch.Tensor, y: torch.Tensor, is_train_set=True) -> None:
        """Expects to be run within 'with torch.no_grad()'"""
        self.net.eval()

        train_loss = self.criterion(self.net(x), y)

        if is_train_set:
            self.losses.append(train_loss)
            print(f"train loss: {train_loss:.3f}")
            self.time_steps.append(self.time_step)
        else:
            val_loss = self.criterion(self.net(x), y)
            print(f"val loss: {val_loss:.3f}")

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

                if not minibatch_i % evaluate_train_every:
                    self.evaluate(x=x, y=y)

                if evaluate_val_every:
                    if not minibatch_i % evaluate_val_every:
                        self.evaluate(x=x_val, y=y_val, is_train_set=False)

            self.time_step += 1

    def fit(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, n_epochs: int, lr: float,
            x_val: torch.Tensor, y_val: torch.Tensor, optimizer=torch.optim.Adam,
            evaluate_each_epoch=True, evaluate_train_every: int = 10, evaluate_val_every: int = 0
            ) -> None:

        optimizer: torch.optim.Adam = optimizer(self.net.parameters(), lr=lr)

        self.time_step = 0
        self.losses = []
        self.val_losses = {}
        self.time_steps = []

        for _ in range(n_epochs):
            self.run_epoch(x, y, x_val, y_val, batch_size=batch_size, optimizer=optimizer,
                           evaluate_train_every=evaluate_train_every,
                           evaluate_val_every=evaluate_val_every)

            if evaluate_each_epoch:
                self.evaluate(x=x_val, y=y_val, is_train_set=False)


dp_slicer = slice(5_000)  # N data points to use
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
trainer = NetTrainer(num_classes=4032)
trainer.fit(x=X, y=Y, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR, x_val=X_val, y_val=Y_val)

# Plot loss over time
plt.plot(trainer.time_steps, trainer.losses)
plt.plot(*list(zip(*trainer.val_losses.items())))
plt.ylim(min(trainer.losses) * 0.8, 0.04)  # Zoom in on where the action's at
plt.show()

legal_moves_net: ResNet
legal_moves_net.eval()


def calculte_correct(in_x, in_y, n_top):
    # Evaluate accuracy and correctness
    preds: torch.Tensor = legal_moves_net(in_x).sigmoid()

    y_densed = defaultdict(list)
    for i_ in np.stack(np.where(in_y), axis=1):
        y_densed[i_[0]].append(i_[1])

    preds_topx = preds.sort(dim=1)[1][:, -n_top:]

    score = 0
    for i_, topx in enumerate(preds_topx):
        if all([t in y_densed[i_] for t in topx]):
            score += 1
    return score / in_x.shape[0]


print("Correct train: ", calculte_correct(X, Y, 10))
print("Correct test: ", calculte_correct(X_val, Y_val, 10))

# Plot output vs actual labels for a single point
i = np.random.randint(0, X_val.shape[0])
est = legal_moves_net(X[i:i + 1]).sigmoid().detach().numpy().flatten()
a = np.zeros(4032)
a[np.where(Y[i])[0]] = est.max()

plt.close('all')
plt.plot(np.arange(len(a)), a)
plt.plot(np.arange(4032), est)
plt.show()

# todo
#  - Check top 5, 10, 15 outputs and see if they're in legal moves
#  - Consider using fastai library
#  - Set it up so that it can train on (almost?) all of the available games
