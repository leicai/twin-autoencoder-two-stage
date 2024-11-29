import torch
import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(indim, indim * 2),
            nn.LeakyReLU(),
            nn.Linear(indim * 2, indim * 4),
            nn.LeakyReLU(),
            nn.Linear(indim * 4, outdim),
        )

    def forward(self, x):
        layer_out = self.layers(x.float())
        linear_out = self.linear(layer_out)
        return layer_out, linear_out


class CNN_model(object):
    def __init__(self, num_epoches, learning_rate, every_epoch_print, device, indim, outdim):
        self.device = device
        self.model = CNN(indim, outdim).to(device)
        self.num_epoches = num_epoches
        self.learning_rate = learning_rate
        self.every_epoch_print = every_epoch_print

    def fit(self, trainX, trainY):
        criterion1 = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-8)
        trainX = trainX.to(self.device)
        trainY = trainY.to(self.device)
        # GPU is available?
        for epoch in range(self.num_epoches):
            layer_out, outputs = self.model(trainX)
            optimizer.zero_grad()
            loss = criterion1(outputs, trainY)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            if epoch % self.every_epoch_print == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        return self.model

    def predict(self, X_test):
        X_test = X_test.to(self.device)
        return self.model(X_test)[1].to('cpu')

    def get_name(self):
        return "CNN_model"
