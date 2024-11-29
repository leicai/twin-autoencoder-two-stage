import torch.nn as nn
import argparse
import torch
from torch.autograd import Variable
from sklearn import metrics
import numpy as np
from config import cfg_from_yaml_file, cfg
from torch.nn.modules.distance import PairwiseDistance      # L2 欧式距离
import matplotlib.pyplot as plt
# from tslearn.metrics import dtw
from dtw import *
from fastdtw import fastdtw


class DLoss(nn.Module):
    """
        仿照 MSELoss 计算两条曲线之间的距离放入 Loss
    """
    def __init__(self):
        super(DLoss, self).__init__()
        # self.deta = deta

    def forward(self, source, target):
        res = ((source - target) * 100) ** 2
        loss = res.sum(axis=(0, 1)) / len(source[0])
        return loss


class FittedLoss(nn.Module):
    """
        仿照 MSELoss 计算 output 和 fitted curve 之间的距离放入 Loss
    """
    def __init__(self):
        super(FittedLoss, self).__init__()
        # self.deta = deta

    def forward(self, source, target):
        # get shape = (166) curve
        source = source.transpose(0, 1)  # 更换 0 - 1 轴
        source = source.mean(axis=1)
        res = (source - target) ** 2
        loss = res.mean(axis=0)
        return loss


class FastDTWLoss(torch.nn.Module):
    """
    计算两个时间序列之间的 fastDTW 距离, 调用 FastDTW
    """

    def __init__(self, radius=1):
        super(FastDTWLoss, self).__init__()
        self.radius = radius

    def forward(self, source, target):
        """
        :param source: 第一个时间序列，类型为 torch.Tensor
        :param target: 第二个时间序列，类型为 torch.Tensor
        :return: fastDTW 距离
        """
        source = source.squeeze().detach().numpy()
        target = target.squeeze().detach().numpy()
        distance, _ = fastdtw(source, target, radius=self.radius)
        distance = torch.tensor(distance, dtype=torch.float32)
        return distance



class DTWLoss2(torch.nn.Module):
    def __init__(self):
        super(DTWLoss2, self).__init__()

    def forward(self, source, target):
        source = source.squeeze().numpy()
        target = target.squeeze().numpy()

        m, n = len(source), len(target)

        DTW = np.zeros((m+1, n+1))
        DTW[:, 0] = np.inf
        DTW[0, :] = np.inf
        DTW[0, 0] = 0

        for i in range(1, m+1):
            for j in range(1, n+1):
                dist = (source[i-1] - target[j-1])**2
                DTW[i, j] = dist + np.min([DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1]])

        loss = torch.sqrt(torch.Tensor([DTW[m, n]]))

        return loss


class DTWLoss1(nn.Module):
    """
        计算两个时间序列之间的 DTW 距离
    """
    def __init__(self):
        super(DTWLoss1, self).__init__()

    def forward(self, source, target):
        """
        :param source: 第一个时间序列，类型为 torch.Tensor
        :param target 第二个时间序列，类型为 torch.Tensor
        :return: DTW 距离
        """
        source = source.squeeze()
        target = target.squeeze()
        DTW = {}
        for i in range(len(source)):
            DTW[(i, -1)] = float('inf')
        for i in range(len(target)):
            DTW[(-1, i)] = float('inf')
        DTW[(-1, -1)] = 0

        for i in range(len(source)):
            for j in range(len(target)):
                dist = (source[i] - target[j]) ** 2
                DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
        loss =  torch.sqrt(DTW[len(source) - 1, len(target) - 1])
        print(loss)
        return loss


class DTWLoss(nn.Module):
    """
        计算两个时间序列之间的 DTW 距离
    """
    def __init__(self):
        super(DTWLoss, self).__init__()

    def forward(self, source, target):
        """
        :param source: 第一个时间序列，类型为 torch.Tensor
        :param target 第二个时间序列，类型为 torch.Tensor
        :return: DTW 距离
        """
        source = source.squeeze()
        target = target.squeeze()
        m = len(source)
        n = len(target)
        dtw = np.zeros((m+1, n+1))
        for i in range(1, m+1):
            for j in range(1, n+1):
                cost = abs(source[i-1] - target[j-1])
                dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
        loss = dtw[m][n]
        print(loss)
        return loss


class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        if r2.item() < 0:
            r2 = torch.tanh(r2) * -2
        else:
            r2 = 1 - torch.tanh(r2)
        return r2


# class FittedLoss(nn.Module):
#     """
#         考虑向拟合曲线的过拟合, 使用电压数据的老化情况加以限制, 缩小 Fitted Loss
#     """
#     def __init__(self):
#         super(FittedLoss, self).__init__()
#         # self.deta = deta
#
#     def forward(self, source, target, offset, cycle_num):
#         # get shape = (166) curve
#         source = source.transpose(0, 1)  # 更换 0 - 1 轴
#         source = source.mean(axis=1)
#         res = (source - target) ** 2
#         loss = res.mean(axis=0)
#
#         # loss delay
#         source_offset = getOffset(cycle_num, source)
#         # deta will be increase while source_offset close to offset
#         t = source_offset - offset
#         deta = abs(source_offset / offset)
#         # print(t)
#
#         # if offset_loss increase, the FittedLoss will be decrease
#         # loss = loss * (1 - deta)
#         return loss


# class OffsetLoss(nn.Module):
#     """
#         仿照 MSELoss 计算 output 和 fitted curve 之间的距离放入 Loss
#     """
#     def __init__(self):
#         super(OffsetLoss, self).__init__()
#         # self.deta = deta
#
#     def forward(self, source, offset):
#         # calc source offset
#         # get shape = (166) curve
#         source = source.transpose(0, 1)  # 更换 0 - 1 轴
#         source = source.mean(axis=1)
#         source_offset = getOffset(50, source)
#         res = (offset - source_offset) ** 2
#         loss = res.mean(axis=0)
#         return loss




