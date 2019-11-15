from collections import defaultdict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from resnet_module import ResNet, BasicBlock


def run_epoch(batch_x: torch.Tensor, batch_y: torch.Tensor, optimizer, criterion, net, batch_size,
              losses, time_steps, val_losses):
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

                evaluate(in_net=net, in_val_x=val_X, in_val_y=val_y, criterion=criterion,
                         val_losses=val_losses, time_steps=time_steps)

                print(f"train loss: {train_loss:.3f}"
                      f", val loss: {list(val_losses.values())[-1]:.3f}")


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
    model.train()
    model.cpu()
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    val_losses = {}
    time_steps = [0]
    for _ in range(n_epochs):
        run_epoch(batch_x, batch_y, optimizer=optimizer, criterion=criterion, net=model,
                  batch_size=batch_size, losses=losses, time_steps=time_steps,
                  val_losses=val_losses)

        # evaluate(in_net=model, in_val_x=val_X, in_val_y=val_y, criterion=criterion,
        #          val_losses=val_losses, time_steps=time_steps)

        # print(f"train loss: {losses[-1]:.3f}"
        #       f", val loss: {list(val_losses.values())[-1]:.3f}")

    return model, losses, time_steps, val_losses


dp_slicer = slice(100_000)  # N data points to use
N_EPOCHS = 2
LR = 1e-3
BATCH_SIZE = 128

# Load a teporarily saved sample data
X = torch.tensor(joblib.load("./tmp_batch_x")).float()[dp_slicer]
y = torch.tensor(joblib.load("./tmp_batch_y"))[dp_slicer].float()
fens = joblib.load("./tmp_batch_fens")[dp_slicer]

val_X = torch.tensor(joblib.load("./tmp_val_x")).float()
val_y = torch.tensor(joblib.load("./tmp_val_y")).float()
val_fens = joblib.load("./tmp_val_fens")

print(X.shape[0], val_X.shape[0])
legal_moves_net, train_losses, ts, val_l = fit(batch_x=X, batch_y=y, batch_size=BATCH_SIZE,
                                               n_epochs=N_EPOCHS, lr=LR)

# Plot loss over time
plt.plot(ts[1:], train_losses)
plt.plot(*list(zip(*val_l.items())))
plt.ylim(min(train_losses) * 0.8, 0.04)  # Zoom in on where the action's at
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


print("Correct train: ", calculte_correct(X, y, 10))
print("Correct test: ", calculte_correct(val_X, val_y, 10))

# Plot output vs actual labels for a single point
i = np.random.randint(0, val_X.shape[0])
est = legal_moves_net(X[i:i + 1]).sigmoid().detach().numpy().flatten()
a = np.zeros(4032)
a[np.where(y[i])[0]] = est.max()

plt.close('all')
plt.plot(np.arange(len(a)), a)
plt.plot(np.arange(4032), est)
plt.show()

# todo
#  - Check if evaluation function works properly
#  - Check top 5, 10, 15 outputs and see if they're in legal moves
#  - Consider using fastai library
#  - Set it up so that it can train on (almost?) all of the available games
