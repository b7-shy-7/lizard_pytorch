import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FashionMNIST
from torch.

from network import Net

batch_size = 100

data = FashionMNIST(
    root = './data',
    train = True,
    transform = transform.Compose([
        transform.ToTensor()
    ]),
    download = True
)

net = Net(batch_size=batch_size).cuda()

loader = DataLoader(data, batch_size = batch_size)
optimizer = optim.Adam(net.parameters(), lr = 0.01)
loss = nn.CrossEntropyLoss().cuda()

correct_number = 0

for i in range(10):
    for data, label in loader:
        data = data.cuda()
        label = label.cuda()
        output = net(data)
        label = label
        # output = torch.argmax(output, dim = 1)
        loss = nn.CrossEntropyLoss()(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct_number += torch.argmax(output, dim = 1).eq(label).sum(dim = 0)
        # print(loss)
    print(correct_number.item())
    correct_number = 0

