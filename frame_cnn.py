import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
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
                nn.Sigmoid())

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

class FrameCnnDataset(D.Dataset):
    def __init__(
            self,
            size,
            world_size,
            valid=False):
        # TODO Make sure there aren't duplicates
        self.data = [
                W.make_world(world_size)
                for _ in range(size)
                ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    

def main():
    model = FrameCnn()
    metric = nn.MSELoss()
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5)

    num_worlds = 10000
    world_size = 24
    batch_size = 10
    dataset = FrameCnnDataset(num_worlds, world_size)
    test_size = len(dataset) // 5
    train_ds, test_ds = D.random_split(
            dataset,
            [len(dataset) - test_size, test_size])
    train_dl = D.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True)
    test_dl = D.DataLoader(
            test_ds,
            batch_size=batch_size)

    num_epochs = 15
    for epoch in range(num_epochs):
        for batch in train_dl:
            output = model(batch)
            loss = metric(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch [{epoch+1}/{num_epochs}], '
              f'loss: {loss:.4f}')

    avg_loss = 0
    for batch in test_dl:
        output = model(batch)
        if avg_loss == 0: # shorthand for first batch
            for i in range(len(batch)):
                print(W.show_fancy(batch[i]))
                print(W.show_fancy(output[i]))
                print()
        loss = metric(output, batch)
        avg_loss += loss * len(batch)/(len(test_dl) * test_dl.batch_size)

    print(f'test loss: {avg_loss:.4f}')
    torch.save(model.state_dict(), './frame_cnn.pth')


if __name__ == '__main__':
    main()
