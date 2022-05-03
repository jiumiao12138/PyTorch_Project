# import torch
# import torch.nn.functional as F
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]])
# kernel = torch.tensor([[1, 2, 1],
#                        [0, 1, 0],
#                        [2, 1, 0]])
#
# input = torch.reshape(input, (1, 1, 5, 5))
# kernel = torch.reshape(kernel, (1, 1, 3, 3))
#
# print(input.shape)
# print(kernel.shape)
#
# output1 = F.conv2d(input, kernel, stride=1)
# print(output1)
#
# output2 = F.conv2d(input, kernel, stride=2)
# print(output2)
#
# output3 = F.conv2d(input, kernel, stride=1, padding=1)
# print(output3)

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset_learn", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

model = Model()

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])  -> [xxx, 3, 30, 30]

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step = step + 1

writer.close()