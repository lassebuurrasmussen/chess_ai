import chess
import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from tqdm import tqdm

import utility_module as ut


class ReluConv(nn.Module):
    """A convolution with relu activation"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ReluConv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.maxpooling = nn.MaxPool2d(3, padding=padding)

    def forward(self, x):
        return F.relu(self.conv(x))


class LegalMovePredictor(nn.Module):
    def __init__(self):
        super(LegalMovePredictor, self).__init__()
        out_channels = 3
        self.conv1 = ReluConv(in_channels=12, out_channels=out_channels, kernel_size=3, stride=1,
                              padding=1)

        n_moves = 4032
        self.classification = nn.Linear(out_channels * 8 * 8, n_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        downsample = self.conv1(x)
        flat = downsample.view(len(downsample), -1)
        output = F.softmax(self.classification(flat), 1)

        return output


def run_epoch(batch_x: torch.Tensor, batch_y: torch.Tensor, optimizer, criterion, net, batch_size,
              losses):
    start_idxs = range(0, batch_x.shape[0], batch_size)
    for minibatch_i, start_idx in enumerate(start_idxs):
        optimizer.zero_grad()

        slicer = slice(start_idx, start_idx + batch_size)

        output = net(batch_x[slicer])

        loss = criterion(output, batch_y[slicer])
        loss.backward()

        with torch.no_grad():
            optimizer.step()

            if not minibatch_i % 10:
                train_loss = criterion(net(batch_x), batch_y)
                losses.append(train_loss)


def fit(batch_x, batch_y, batch_size=64 * 2, n_epochs=70, lr=1e-3):
    lmp: LegalMovePredictor = LegalMovePredictor()
    lmp.float()
    optimizer: torch.optim.Adam = torch.optim.Adam(lmp.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for _ in tqdm(range(n_epochs)):
        run_epoch(batch_x, batch_y, optimizer=optimizer, criterion=criterion, net=lmp,
                  batch_size=batch_size, losses=losses)

    return lmp, losses


dp_slicer = slice(200)  # N data points to use
N_EPOCHS = 170
LR = 1e-3

# Load a teporarily saved sample data
X = torch.tensor(joblib.load("./tmp_batch_x")).float()[dp_slicer]
y = torch.tensor(joblib.load("./tmp_batch_y"))[dp_slicer]
fens = joblib.load("./tmp_batch_fens")[dp_slicer]

legal_moves_net, train_losses = fit(batch_x=X, batch_y=y, n_epochs=N_EPOCHS, lr=LR)

# Plot loss over time
plt.plot(range(len(train_losses)), train_losses)
plt.show()

# Evaluate in-sample accuracy and correctness
preds = legal_moves_net(X).argmax(1)
accuracy = (preds == y).int().float().mean()

correct = 0
for i, fen in enumerate(fens):
    if preds[i] in [ut.uci2onehot_idx(str(uci)) for uci in chess.Board(fen).legal_moves]:
        correct += 1

print(f"Accuracy: {accuracy:.3f}")
print(f"Prediction correct ratio: {correct / len(fens):.3f}")
