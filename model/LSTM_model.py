import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.losses.DLoss import R2Loss, FittedLoss

'''
    添加了 Dropout, 如果不需要, 关闭即可
'''
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        # add
        self.dp = nn.Dropout(p=0.05)
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=int(hidden_size/2), kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=int(hidden_size/2), out_channels=int(hidden_size/4), kernel_size=1)
        self.actfunc = nn.LeakyReLU()

        self.fc1 = nn.Linear(int(hidden_size/4), int(hidden_size/8))
        self.fc2 = nn.Linear(int(hidden_size/8), num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.dp(h_out)  # add

        # 卷积
        out = out.transpose(0, 1)
        out = self.conv1(out)  # add
        out = self.actfunc(out)
        out = self.conv2(out)  # add
        out = self.actfunc(out)

        # 全连接
        out = out.transpose(0, 1)
        out = self.fc1(out)
        out = self.actfunc(out)
        out = self.fc2(out)
        return out


class LSTM_model(object):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, num_epoches, learning_rate, every_epoch_print):
        self.lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
        self.num_epoches = num_epoches
        self.learning_rate = learning_rate
        self.every_epoch_print = every_epoch_print

    def fit(self, trainX, trainY, trainZ):
        criterion1 = torch.nn.MSELoss()
        criterion2 = R2Loss()
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)

        weight_coefficient = 0.998
        for epoch in range(self.num_epoches):
            outputs = self.lstm(trainX)
            # print(outputs.shape)
            optimizer.zero_grad()
            loss1 = criterion1(outputs, trainY)
            loss2 = criterion2(outputs, trainZ)
            # 需要保证 R2 的系数足够小, 否则loss修正效果过大会引起其失效
            loss = weight_coefficient * loss1 + (1 - weight_coefficient) * loss2
            # loss = loss1
            loss.backward()
            optimizer.step()
            if epoch % self.every_epoch_print == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
                print("     loss1: %1.5f, loss2: %1.5f" % (loss1.item(), loss2.item()))
        return self.lstm

    def predict(self, X_test):
        return self.lstm(X_test)

    def get_name(self):
        return "LSTM_Model"
