import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from network import Net

batch_size = 100

dataset = FashionMNIST(
    root = './data',
    train = True,
    transform = transform.Compose([
        transform.ToTensor()
    ]),
    download = True
)

net = Net(batch_size=batch_size)

loader = DataLoader(dataset, batch_size = batch_size)
optimizer = optim.Adam(net.parameters(), lr = 0.01)
loss = nn.CrossEntropyLoss()

correct_number = 0

tb  = SummaryWriter()
images, labels = next(iter(loader))
grid = torchvision.utils.make_grid(images)

tb.add_image('images', grid)
tb.add_graph(net, images)
# tb.close()

for epoch in range(10):
    total_loss = 0
    total_correct = 0
    for data, label in tqdm(loader, desc = "Training Epoch %d: ".format(epoch)):
        # data = data.cuda()
        # label = label.cuda()
        output = net(data)
        label = label
        # output = torch.argmax(output, dim = 1)
        loss = nn.CrossEntropyLoss()(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct_number += torch.argmax(output, dim = 1).eq(label).sum(dim = 0)

        total_loss += loss.item()
        total_correct = correct_number
        # print(loss)
    
    tb.add_scalar('Loss', total_loss, epoch)
    tb.add_scalar('Number Correct', total_correct, epoch)
    tb.add_scalar('Accuracy', correct_number.item() / len(dataset), epoch)

    tb.add_histogram('conv1.bias', net.conv1.bias, epoch)
    tb.add_histogram('conv.weight', net.conv1.weight, epoch)
    tb.add_histogram('conv.weight.grad', net.conv1.weight.grad, epoch)

    # print(correct_number.item())
    # correct_number = 0

