import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


import world as W

"""
(24, 1) - input
(24, 6) - 3 conv, 6 filters
(12, 6) - 2 max poooling
(12, 6) - 3 conv, 6 filters
(6, 6) - 2 max pooling
(36,) - flatten
(36,) - fc
(6,) - fc
"""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(nn.Module):
    def __init__(self, *new_shape):
        super(Unflatten, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return x.view(x.size(0), *self.new_shape)

class FrameCnn(nn.Module):
    def __init__(self):
        super(FrameCnn, self).__init__()
        
        # Batch normalization?
        self.encoder1 = nn.Sequential(
                nn.Conv1d(1, 6, 3, padding=1),
                nn.ReLU(True),
                nn.MaxPool1d(2, return_indices=True))
        self.encoder2 = nn.Sequential(
                nn.Conv1d(6, 6, 3, padding=1),
                nn.ReLU(True),
                nn.MaxPool1d(2, return_indices=True))
        self.encoder3 = nn.Sequential(
                Flatten(),
                nn.Linear(36, 36),
                nn.ReLU(True),
                nn.Linear(36, 6))

        self.decoder1 = nn.Sequential(
                nn.Linear(6, 36),
                nn.ReLU(True),
                nn.Linear(36, 36),
                Unflatten(6, 6),)
        self.maxunpool1 = nn.MaxUnpool1d(2)
        self.decoder2 = nn.Sequential(
                nn.ConvTranspose1d(6, 6, 3, padding=1),
                nn.ReLU(True),) # a second relu?
        self.maxunpool2 = nn.MaxUnpool1d(2)
        self.decoder3 = nn.Sequential(
                nn.ConvTranspose1d(6, 1, 3, padding=1),
                nn.Tanh())

    def forward(self, x):
        x, mp_indices1 = self.encoder1(x)
        x, mp_indices2 = self.encoder2(x)
        x = self.encoder3(x)
        x = self.decoder1(x)
        x = self.maxunpool1(x, mp_indices2)
        x = self.decoder2(x)
        x = self.maxunpool2(x, mp_indices1)
        x = self.decoder3(x)
        return x

'''
TODO
- Generate test set
- Visualization
- Dropout/sparsity/denoising
- Dataset shuffling
'''

class FrameCnnDataset(Dataset):
    def __init__(
            self,
            size,
            world_size,
            test_prop=8,
            batch_size=10,
            valid=False):

        self.data = [
                W.make_world(world_size)
                for _ in range(size)
                ]
        if valid:
            pass
        else:
            self.valid = []

    def shuffle_existing(old_ds):
        # Take an existing dataset, shuffle the train, test, and valid
        # and return a new one.
        pass

    def __len__(self):
        return len(self.train) + len(self.test) + len(self.valid)

    def __getitem__(self, i):
        pass

    

def main():
    model = FrameCnn()
    metric = nn.MSELoss()
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5)

    num_batches = 100
    batch_size = 10
    world_size = 24
    worlds = [
            W.make_world_batch(world_size, batch_size)
            for _ in range(num_batches)
            ]

    num_epochs = 40
    for epoch in range(num_epochs):
        for world in worlds:
            output = model(world)
            loss = metric(output, world)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch [{epoch+1}/{num_epochs}], loss: {loss:.4f}')

    torch.save(model.state_dict(), './frame_cnn.pth')


if __name__ == '__main__':
    main()
