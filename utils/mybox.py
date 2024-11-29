import argparse
from config import cfg_from_yaml_file, cfg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_config(filepath):
    """
    :param filepath: config .xml file path
    :return: cfg, a dictionary
    :example:
        args = read_config('cfgs/MIT_configs/base_config.yaml')
    """
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_file', type=str, default=filepath)
    args = parser.parse_args()
    cfg_from_yaml_file(args.model_file, cfg)
    return cfg


def parse_MIT(index):
    """
    :param index: clustering index
    :return: cell names in this cluster
    :desc: get MIT cluster :index all cell name
    """
    cfg = read_config('cfgs/MIT_configs/base_config.yaml')
    categories = pd.read_csv(cfg.data_path + "categories.csv", header=None).values
    part = [[], [], [], []]
    indexs = [[], [], [], []]
    i = 0
    for item in categories:
        part[item[1]].append('cell' + str(int(item[0])))
        indexs[item[1]].append(i)
        i += 1
    return np.asarray(part[index]), np.asarray(indexs[index])


def myplot(xlabel, ylabel, title, grid=True):
    """
    :param xlabel:
    :param ylabel:
    :param grid:
    :param title:
    :return:
    :desc: Unify and simplify drawing format
    """
    # 绘图
    plt.figure(figsize=(9, 5), dpi=150)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)
    if grid:
        plt.grid(color='gray', ls='-.', lw=0.25)
    plt.title(title)
    return plt


def find_bolder(cluster: int):
    """
    :param cluster: choose a cluster in 0,1,2
    :return: [lower, closet, upper cell indexs] and [combine capacity matrix]
    """
    args = read_config('cfgs/MIT_configs/base_config.yaml')
    _strs = pd.read_csv(args.data_path + "\\closet.csv", header=None).values[:, 0]
    closet = 'cell' + str(_strs[int(cluster)])
    # 找到聚类中心距离最远的两条曲线, 作为上下界
    cells, _ = parse_MIT(cluster)

    result = []
    for cell in cells:
        temp = []
        capacity = pd.read_csv(args.data_path + cell + "\\capacity.csv", header=None).values
        temp.append(cell)
        temp.append(capacity.shape[0])
        result.append(temp)
    result = np.asarray(result)
    r = result[:, 1].astype(float)
    result = result[np.argsort(r)]
    np.set_printoptions(suppress=True)

    mins_cell = result[0][0]
    maxs_cell = result[result.shape[0]-1][0]

    lower_c = pd.read_csv(args.data_path + mins_cell + "\\capacity.csv", header=None).values
    closet_c = pd.read_csv(args.data_path + closet + "\\capacity.csv", header=None).values
    upper_c = pd.read_csv(args.data_path + maxs_cell + "\\capacity.csv", header=None).values

    matrix = []
    matrix.append(lower_c.flatten())
    matrix.append(closet_c.flatten())
    matrix.append(upper_c.flatten())

    indics = []
    indics.append(mins_cell)
    indics.append(closet)
    indics.append(maxs_cell)

    # 绘图部分
    # plt = myplot(xlabel='x', ylabel='y', title='Bolder')
    # for cell in cells:
    #     capacity = pd.read_csv(args.data_path + cell + "\\capacity.csv", header=None).values.flatten()
    #     if cell == mins_cell or cell == maxs_cell:
    #         plt.plot(capacity, c='r', lw=1)
    #     elif closet == cell:
    #         plt.plot(capacity, c='b', lw=1.2)
    #     else:
    #         plt.plot(capacity, c='gray', lw=0.7)
    # # plt.legend(loc='best', fontsize=10)
    # plt.show()
    return indics, matrix


def reshape_c(curve, lens):
    """
    :param curve: 需要插值|删值的曲线
    :param lens: 插值|删值的长度
    :return:
    """
    # 插值 | 删值
    output = np.interp(np.linspace(0, len(curve) - 1, lens), np.arange(len(curve)), curve)
    return output.tolist()


def mapping_c(curves, mins, maxs):
    """
    :param curve: 待映射的曲线
    :param mins: 最小的的映射 cycle
    :param maxs: 最大的映射 cycle
    :return: 映射完毕、cycle 数递增的老化曲线
    """
    # 处理曲线的 cycles 分布, 删值 - 处理的意义
    size = len(curves)
    outputs = []
    for i in range(size):
        curve = curves[i]
        lens_ = (mins + int(i / size * (maxs - mins)))
        curve = reshape_c(curve, lens_)  # 删值
        outputs.append(curve)
    return outputs

