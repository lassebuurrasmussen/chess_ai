import chess
import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# noinspection PyPep8Naming
from tqdm import tqdm

import utility_module as ut
from resnet_module import ResNet, BasicBlock


def run_epoch(batch_x: torch.Tensor, batch_y: torch.Tensor, optimizer, criterion, net, batch_size,
              losses, time_steps):
    start_idxs = range(0, batch_x.shape[0], batch_size)
    for minibatch_i, start_idx in tqdm(list(enumerate(start_idxs))):
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
                time_steps.append(minibatch_i + 1 + time_steps[-1])


def evaluate(in_net, in_val_x, in_val_y, criterion, val_losses, time_steps):
    in_net.eval()

    with torch.no_grad():
        in_net(in_val_x)
        val_loss = criterion(in_net(in_val_x), in_val_y)
        val_losses[time_steps[-1]] = val_loss

    in_net.train()


def fit(batch_x, batch_y, batch_size, n_epochs, lr):
    model: ResNet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=4032)
    model.float()
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    val_losses = {}
    time_steps = [0]
    for _ in range(n_epochs):
        run_epoch(batch_x, batch_y, optimizer=optimizer, criterion=criterion, net=model,
                  batch_size=batch_size, losses=losses, time_steps=time_steps)

        # if not epoch % 10:
        evaluate(in_net=model, in_val_x=val_X, in_val_y=val_y, criterion=criterion,
                 val_losses=val_losses, time_steps=time_steps)

    return model, losses, time_steps, val_losses


dp_slicer = slice(10_000)  # N data points to use
N_EPOCHS = 2
LR = 1e-3
BATCH_SIZE = 128

# Load a teporarily saved sample data
X = torch.tensor(joblib.load("./tmp_batch_x")).float()[dp_slicer]
y = torch.tensor(joblib.load("./tmp_batch_y"))[dp_slicer]
fens = joblib.load("./tmp_batch_fens")[dp_slicer]

val_X = torch.tensor(joblib.load("./tmp_val_x")).float()
val_y = torch.tensor(joblib.load("./tmp_val_y"))
val_fens = joblib.load("./tmp_val_fens")

legal_moves_net, train_losses, ts, val_l = fit(batch_x=X, batch_y=y, batch_size=BATCH_SIZE,
                                               n_epochs=N_EPOCHS, lr=LR)

# Plot loss over time
plt.plot(ts[1:], train_losses)
plt.plot(*list(zip(*val_l.items())))
plt.show()

# Evaluate accuracy and correctness
preds = legal_moves_net(X).argmax(1)
accuracy = (preds == y).int().float().mean()

preds_val = legal_moves_net(val_X).argmax(1)
accuracy_val = (preds_val == val_y).int().float().mean()

correct = 0
for i, fen in enumerate(fens):
    if preds[i] in [ut.uci2onehot_idx(str(uci)) for uci in chess.Board(fen).legal_moves]:
        correct += 1

correct_val = 0
for i, fen in enumerate(val_fens):
    if preds_val[i] in [ut.uci2onehot_idx(str(uci)) for uci in chess.Board(fen).legal_moves]:
        correct_val += 1

print(f"Train accuracy: {accuracy:.3f}")
print(f"Train prediction correct ratio: {correct / len(fens):.3f}")

print(f"Test accuracy: {accuracy_val:.3f}")
print(f"Test prediction correct ratio: {correct_val / len(val_fens):.3f}")
