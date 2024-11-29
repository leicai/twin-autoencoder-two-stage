import unittest
import argparse
import pandas as pd
from model.CAEModel import CNNAEFeature
from config import cfg_from_yaml_file, cfg
import numpy as np
import torch
from utils.mybox import read_config, myplot, parse_MIT
from torch.autograd import Variable


class TestGenerateCase(unittest.TestCase):

    def setUp(self) -> None:
        # 这里传递引用 yaml 配置文件的意义在于, 当某些参数需要自定义改变时
        # 便于修改, 不需要在代码里寻找并改动
        self.args = self._read_config()
        self.num_epochs = self.args.CAE.num_epochs
        self.batch_size = self.args.CAE.batch_size
        self.learning_rate = self.args.CAE.learning_rate
        self.every_epoch_print = self.args.CAE.every_epoch_print
        self.feature_num = self.args.CAE.feature_num    # encoder 输出特征大小
        self.device = self.args.CAE.device
        self.gen_num = self.args.CAE.gen_num        # 生成曲线的数目
        self.expand = 20       # 两条曲线数据太少, 扩展几条用于训练

    def _read_config(self):
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--model_file', type=str, default='cfgs/MIT_configs/model_config.yaml')
        args = parser.parse_args()
        cfg_from_yaml_file(args.model_file, cfg)
        return cfg

    def test_generator(self) -> None:
        colors = ['pink', 'plum', 'red', 'seagreen', 'silver', 'snow', 'teal', 'orange', 'yellow', 'blue', 'black']
        # 防止模型训练结果无法复现, 设置种子
        torch.manual_seed(1)  # 为CPU设置随机种子
        torch.cuda.manual_seed_all(1)  # 为所有GPU设置随机种子
        # 取 part = 1 部分的所有曲线
        cell_name = parse_MIT(1)
        voltages = []
        capacities = []
        mins = 100000
        maxs = -1
        for item in cell_name:
            # voltage = pd.read_csv(self.args.data_path + "cell" + str(item) + "\\voltage.csv", header=None).values
            # voltages.append(voltage.tolist())
            capacity = pd.read_csv(self.args.data_path + item + "\\capacity.csv", header=None).values[0 : 325]
            mins = min(mins, capacity.shape[0])
            maxs = max(maxs, capacity.shape[0])
            # capacities.append(capacity.flatten().tolist())
            capacities.append(capacity.tolist())
        capacities = np.asarray(capacities)
        plt = myplot(xlabel='x', ylabel='y', title= 'Part Curve')
        for i in range(capacities.shape[0]):
            plt.plot(capacities[i], c = colors[i%8], lw = 0.8)
        plt.legend(loc='best', fontsize=10)
        plt.show()

        capacities = capacities.swapaxes(1, 2)
        # print(capacities)
        # 扩展到 3 维

        model = CNNAEFeature(capacities.shape[1],
                             capacities.shape[2],
                             self.feature_num,
                             self.num_epochs,
                             self.batch_size,
                             self.learning_rate,
                             self.every_epoch_print,
                             self.device)
        traindata = Variable(torch.tensor(capacities)).float()
        model.fit(traindata)  # train process
        output = model.get_decode_capacity(traindata).detach().numpy()
        feature = model.get_feature(traindata)
        # print(output)
        print("feature:", feature.shape)
        print("output:", output.shape)
        output = np.mean(output, axis=1)
        label = np.mean(capacities, axis=1)
        print(output.shape)
        plt = myplot(xlabel='x', ylabel='y', title='CAE Test')
        for i in range(output.shape[0]):
            plt.plot(output[i], c='r', lw=1)
            plt.plot(label[i], c='g', lw=1)
            # plt.plot(output, c='g', label='true', lw=0.8)
        plt.ylim((0.8, 1.2))
        # plt.legend(loc='best', fontsize=10)
        plt.show()

        featue_out = model.get_decoder(feature).detach().numpy()
        featue_out = np.mean(featue_out, axis=1)
        plt = myplot(xlabel='x', ylabel='y', title='Feature Test')
        for i in range(featue_out.shape[0]):
            plt.plot(featue_out[i], c='r', lw=1)
            plt.plot(label[i], c='g', lw=1)
            # plt.plot(output, c='g', label='true', lw=0.8)
        plt.ylim((0.8, 1.2))
        # plt.legend(loc='best', fontsize=10)
        plt.show()





