import random

import joblib
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F

batch_x = torch.tensor(joblib.load("./tmp_batch_x")).float()
batch_y = torch.tensor(joblib.load("./tmp_batch_y")).float().argmax(1)

random.seed(47398)
shuffled_idx = random.sample(range(len(batch_x)), len(batch_x))

batch_x = batch_x[shuffled_idx]
batch_y = batch_y[shuffled_idx]

print(batch_x.shape)


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

    def forward(self, x):
        return F.relu(self.conv(x))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = ReluConv(in_channels=12, out_channels=6, kernel_size=3, stride=1, padding=1)
        # self.conv2 = ReluConv(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.conv3 = ReluConv(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

        # self.fc1 = nn.Linear(8 * 8, 512)
        self.fc1 = nn.Linear(12 * 8 * 8, 512)
        self.classification = nn.Linear(512, 4032)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # downsample = (
        #     self.conv3(
        #         self.conv2(
        #             self.conv1(x)
        #         )))
        # flat = downsample.view(len(downsample), -1)
        flat = x.view(len(x), -1)
        dense = F.relu(self.fc1(flat))

        output = F.softmax(self.classification(dense), 1)

        return output


def run_epoch(x: torch.Tensor, y: torch.Tensor):
    optimizer.zero_grad()

    output = net(x)

    loss = criterion(output, y)
    print(round(loss.item(), 5))

    loss.backward()

    with torch.no_grad():
        optimizer.step()


net: Net = Net()
net.float()
optimizer: torch.optim.Adam = torch.optim.Adam(net.parameters())

criterion = nn.CrossEntropyLoss()

i1 = 2020
i2 = 3030

for _ in range(100):
    run_epoch(batch_x[i1:i2], batch_y[i1:i2])

print(net(batch_x[i1:i2]).argmax(1))
print(batch_y[i1:i2])
