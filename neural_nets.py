import joblib
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


def relu_conv(x, conv, do_relu=True):
    x = conv(x)
    if do_relu:
        x = F.relu(x)
    return x


class ReluConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ReluConv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.maxpooling = nn.MaxPool2d(3, padding=padding)

    def forward(self, x):
        return F.relu(self.conv(x))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ReluConv(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.conv2 = ReluConv(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.conv3 = ReluConv(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

        # self.fc1 = nn.Linear(8 * 8, 512)
        # self.fc1 = nn.Linear(12 * 8 * 8, 512)
        # self.fc1 = nn.Linear(12 * 8 * 8, 1024)
        # self.fc1 = nn.Linear(3 * 8 * 8, 1024)
        # self.fc2 = nn.Linear(512, 1024)
        # self.classification = nn.Linear(1024, 4032)
        self.classification = nn.Linear(3 * 8 * 8, 4032)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        downsample = (
            #     self.conv3(
            #         self.conv2(
            self.conv1(x)
        )

        #         ))
        flat = downsample.view(len(downsample), -1)
        # dense1 = F.relu(self.fc1(x))
        # dense2 = F.relu(self.fc1(flat))
        # dense2 = F.relu(self.fc2(dense1))

        # output = F.softmax(self.classification(dense2), 1)
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
                # print(f"train loss: {round(train_loss.item(), 5)}")
                losses.append(train_loss)


def fit(batch_x, batch_y, batch_size=64 * 2, n_epochs=70, lr=1e-3):
    losses = []
    net: Net = Net()
    net.float()
    optimizer: torch.optim.Adam = torch.optim.Adam(net.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(n_epochs)):
        # print(f"\nEpoch {epoch}")

        run_epoch(batch_x, batch_y, optimizer=optimizer, criterion=criterion, net=net,
                  batch_size=batch_size, losses=losses)

    return net, losses


fraction = slice(50)
N_EPOCHS = 200
# LR = 5e-2
LR = 1e-3

X = torch.tensor(joblib.load("./tmp_batch_x")).float()[fraction]
X = X.view(-1, 12, 8, 8)
y = torch.tensor(joblib.load("./tmp_batch_y"))[fraction]

legal_moves_net, train_losses = fit(batch_x=X, batch_y=y, n_epochs=N_EPOCHS, lr=LR)

plt.plot(range(len(train_losses)), train_losses)
plt.show()

(legal_moves_net(X).argmax(1) == y).int().float().mean()

# Todo - evaluer på alle lovlige moves i stedet for y
