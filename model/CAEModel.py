from abc import ABC
from torch import nn
import torch


class ConvAE(nn.Module):
    def __init__(self, in_channnel_size, linear_size, feature_size, kernel_size):
        super(ConvAE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channnel_size, out_channels=32, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=4, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.Linear(linear_size, int(linear_size/2)),
            nn.LeakyReLU(),
            nn.Linear(int(linear_size/2), 150),
            nn.LeakyReLU(),
            nn.Linear(150, feature_size),
            nn.Dropout(p=0.05)      # 加了 Dropout, 模型 feature 得到的 output 会更平滑一下, 否则震荡很严重
        )

        self.decoder = nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(feature_size, 150),
            nn.LeakyReLU(),
            nn.Linear(150, int(linear_size / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(linear_size / 2), linear_size),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=1, out_channels=4, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=in_channnel_size, kernel_size=kernel_size),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, en_x):
        return self.decoder(en_x)

    def forward(self,x):
        en_x = self.encode(x)
        de_x = self.decode(en_x)
        return de_x

    def get_encode(self, data):
        encode = self.encode(data)
        return encode

    def get_decode(self, data):
        encode = self.encode(data)
        decode = self.decode(encode)
        return decode

    def get_decoder(self, feature):
        return self.decode(feature)


class CNNAEFeature(ABC):
    def __init__(self, in_channnel_size, linear_size, feature_size, num_epochs, batch_size, learning_rate, every_epoch_print, device):
        self.extractor = ConvAE(in_channnel_size, linear_size, feature_size, kernel_size=1)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.every_epoch_print = every_epoch_print
        self.device = device

    def fit(self, capacities):
        capacities = capacities.to(self.device)
        criterion = nn.MSELoss().to(self.device)
        # 权重衰减, 正则化
        optimizer = torch.optim.Adam(self.extractor.parameters(), lr=float(self.learning_rate))
        data_size = capacities.shape[0]
        self.extractor = self.extractor.to(self.device)
        self.extractor.train()
        loss = 0
        for epoch in range(self.num_epochs):
            for i in range(0, data_size, self.batch_size):
                capacity = capacities[i:i + self.batch_size]
                decode = self.extractor(capacity)
                loss = criterion(decode, capacity)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if epoch % self.every_epoch_print == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
                # 提前终止
                loss_1 = loss.item()
                if loss_1 < 0.0000001:
                    break

    def get_feature(self, data):
        return self.extractor.get_encode(data)

    def get_decode_capacity(self, data):
        return self.extractor.get_decode(data)

    def get_decoder(self, feature):
        return self.extractor.get_decoder(feature)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain .net file path
        :return: None
        """
        self.is_fitted = True
        self.extractor.load_state_dict(torch.load(PATH))
        print("Model's state_dict:")
        for param_tensor in self.extractor.state_dict():
            print(param_tensor, "\t", self.extractor.state_dict()[param_tensor].size())

    def save(self,path):
        torch.save(self.extractor.state_dict(), path)



