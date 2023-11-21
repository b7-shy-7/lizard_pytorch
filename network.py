import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

batch_size = 100

class Net(nn.Module):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 3)

        self.fc1 = nn.Linear(in_features = 12 * 5 * 5, out_features = 60)
        self.fc2 = nn.Linear(in_features = 60, out_features = 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size = 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size = 2)
        x = F.relu(x)

        x = x.reshape(self.batch_size, -1)

        x = self.fc1(x)
        return self.fc2(x)
    
if __name__ == '__main__':
    input = torch.tensor(np.random.random((batch_size, 1, 28, 28)), dtype = torch.float32)
    net = Net(batch_size = batch_size)
    output = net(input)
    print(output.shape)