import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import numpy as np
import random
from itertools import islice


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


class Noise(nn.Module):
    def __init__(self, dist):
        super(Noise, self).__init__()
        self.dist = dist

    def forward(self, x):
        if self.training:
            x += self.dist.sample(x.size()) - self.dist.mean
        return x


class FrameCnn(nn.Module):
    def __init__(self):
        super(FrameCnn, self).__init__()

        in_size = (100,)
        in_channels = 1
        conv1_filters = 8
        conv2_filters = 8
        flat_size = conv2_filters * in_size[0] // 4
        fc1_size = 36
        hidden_size = 20
        dropout_rate = 0.3
        self.rohct = 3.0

        self.hidden_size = hidden_size

        # Batch normalization?
        self.encoder1 = nn.Sequential(
            # nn.Dropout(dropout_rate, True),
            # Noise(torch.distributions.normal.Normal(0.0, 0.1)),
            # Noise(torch.distributions.half_normal.HalfNormal(0.05)),
            # Noise(torch.distributions.bernoulli.Bernoulli(0.05)),
            # Noise(torch.distributions.relaxed_bernoulli.RelaxedBernoulli(1,probs=0.1)),
            nn.Conv1d(in_channels, conv1_filters, 5, padding=2),
            nn.ReLU(True),
            nn.MaxPool1d(2),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(conv1_filters, conv2_filters, 5, padding=2),
            nn.ReLU(True),
            nn.MaxPool1d(2),
        )
        self.encoder3 = nn.Sequential(
            Flatten(),
            nn.Linear(flat_size, fc1_size),
            nn.ReLU(True),
            nn.Linear(fc1_size, hidden_size),
            nn.Dropout(dropout_rate, True),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_size, fc1_size),
            nn.ReLU(True),
            nn.Linear(fc1_size, flat_size),
            Unflatten(conv2_filters, in_size[0] // 4),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(
                conv2_filters, conv1_filters, 5, padding=2, stride=2, output_padding=1
            ),
            nn.ReLU(True),
        )  # a second relu?
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(
                conv2_filters, in_channels, 5, padding=2, stride=2, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def apply_rohc(self, x):
        if self.training:
            return torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
                torch.tensor([self.rohct]), logits=x
            ).sample()
        else:
            s = x.shape
            return torch.zeros(s).scatter_(-1, x.argmax(-1).view(s[:-1] + (1,)), 1.0)

    def encode(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        return x


class FCClassifier(nn.Module):
    def __init__(self, frame_cnn, num_classes):
        super(FCClassifier, self).__init__()
        self.frame_cnn = frame_cnn
        self.classifier = nn.Sequential(nn.Linear(frame_cnn.hidden_size, num_classes))
        # Get rid of softmax for training?

    def forward(self, x):
        x = self.frame_cnn.encode(x)
        return self.classifier(x)


def make_datasets(world_size, train_prop=0.8):
    data_dict = W.make_worlds(world_size)
    min_size = min(len(v) for v in data_dict.values())
    print(f"min_size:\t{min_size}")
    print(f"total_size:\t{min_size*len(data_dict)}")
    for k, v in data_dict.items():
        data_dict[k] = random.sample(v, min_size)

    train_size = int(min_size * train_prop)
    train_ds = FrameCnnDataset({k: v[:train_size] for k, v in data_dict.items()})
    test_ds = FrameCnnDataset({k: v[train_size:] for k, v in data_dict.items()})
    return train_ds, test_ds


class FrameCnnDataset(D.Dataset):
    def __init__(self, data_dict):
        """
        Assumes that `data_dict` consists of equal length classes

        """
        self.data_dict = data_dict
        # Make sure we have a consistently ordered list of keys
        self.keys = list(data_dict.keys())
        self.list_len = len(data_dict[self.keys[0]])

    def __len__(self):
        return sum(len(v) for v in self.data_dict.values())

    def __getitem__(self, i):
        sub_collection = self.keys[i // self.list_len]
        sub_index = i % self.list_len
        return self.data_dict[sub_collection][sub_index].to_tensor()


class FCClassifierDataset(D.Dataset):
    def __init__(self, frame_cnn_dataset):
        self.fc_dataset = frame_cnn_dataset
        self.num_classes = len(frame_cnn_dataset.keys)

    def __len__(self):
        return len(self.fc_dataset)

    def __getitem__(self, i):
        # one_hot = torch.zeros((self.num_classes,))
        # one_hot[i // self.fc_dataset.list_len] = 1.0
        i // self.fc_dataset.list_len
        return self.fc_dataset[i], i // self.fc_dataset.list_len


class TwoFrameDataet(D.Dataset):
    def __init__(self, data_dict):
        ...


def main_classifier():
    ae_model = FrameCnn()
    ae_metric = nn.MSELoss()
    # metric = nn.L1Loss()
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)

    world_size = 100  # 24
    batch_size = 20
    # TODO better sampling here; too many examples?
    ae_train_ds, ae_test_ds = make_datasets(world_size, 0.8)
    ae_train_dl = D.DataLoader(ae_train_ds, batch_size=batch_size, shuffle=True)
    ae_test_dl = D.DataLoader(ae_test_ds, batch_size=batch_size, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        losses = []
        for batch in ae_train_dl:
            output = ae_model(batch)
            loss = ae_metric(output, batch)
            losses.append(loss.detach().numpy())
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()

        ae_model.rohct *= 1 - 5e-1
        print(f"epoch [{epoch+1}/{num_epochs}], loss: {np.average(losses):.4f}")
    ae_model.eval()

    avg_loss = 0
    for batch in ae_test_dl:
        output = ae_model(batch)
        if avg_loss == 0:  # shorthand for first batch
           for i in range(len(batch)):
               print(W.show_fancy(batch[i]))
               print(W.show_fancy(output[i]))
               print()
        loss = ae_metric(output, batch)
        avg_loss += loss * len(batch) / (len(ae_test_dl) * ae_test_dl.batch_size)

    print(f"test loss: {avg_loss:.4f}")

    # begin classifier code #

    #for param in ae_model.parameters():
    #    param.requires_grad = False
    ae_model.train()

    train_ds = FCClassifierDataset(ae_train_ds)
    test_ds = FCClassifierDataset(ae_test_ds)

    #ae_model = FrameCnn()
    model = FCClassifier(ae_model, train_ds.num_classes)
    #model = FCClassifier(FrameCnn(), train_ds.num_classes)
    metric = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_dl = D.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = D.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    num_epochs = 1
    for epoch in range(num_epochs):
        losses = []
        for batch in islice(train_dl, len(train_dl) // 2):
            output = model(batch[0])
            loss = metric(output, batch[1])
            losses.append(loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch [{epoch+1}/{num_epochs}], loss: {np.average(losses):.4f}")

    model.eval()
    avg_loss = 0
    avg_acc = 0
    for batch in test_dl:
        output = model(batch[0])
        loss = metric(output, batch[1])
        acc = (output.argmax(dim=-1) == batch[1]).double().mean()
        avg_acc += acc * len(batch[0]) / (len(test_dl) * test_dl.batch_size)
        avg_loss += loss * len(batch[0]) / (len(test_dl) * test_dl.batch_size)

    print(f"test acc: {avg_acc:.4f}\ttest loss: {avg_loss:.4f}")
    # torch.save(model.state_dict(), "./models/frame_cnn.pth")


def main():
    model = FrameCnn()
    metric = nn.MSELoss()
    # metric = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    world_size = 100  # 24
    batch_size = 20
    # TODO better sampling here; too many examples?
    train_ds, test_ds = make_datasets(world_size, 0.8)
    train_dl = D.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = D.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    num_epochs = 40
    for epoch in range(num_epochs):
        losses = []
        for batch in train_dl:
            output = model(batch)
            loss = metric(output, batch)
            losses.append(loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.rohct *= 1 - 5e-1
        print(f"epoch [{epoch+1}/{num_epochs}], loss: {np.average(losses):.4f}")

    model.eval()
    avg_loss = 0
    for batch in test_dl:
        output = model(batch)
        if avg_loss == 0:  # shorthand for first batch
            for i in range(len(batch)):
                print(W.show_fancy(batch[i]))
                print(W.show_fancy(output[i]))
                print()
        loss = metric(output, batch)
        avg_loss += loss * len(batch) / (len(test_dl) * test_dl.batch_size)

    print(f"test loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "./models/frame_cnn.pth")


if __name__ == "__main__":
    # main()
    main_classifier()
