import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import torch
from pathlib import Path
import os
import pandas as pd


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)):
        if i+seq_length == len(data):
            break
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]   # 这里改了，少个0
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def get_end_cycle(series, valid_soh):
    index, _ = np.where(series < valid_soh)
    return index[0]


def clip_window_data(data, seq_length):
    y = []
    for i in range(len(data)):
        if i+seq_length == len(data):
            break
        y.append(data[i+seq_length])
    return np.array(y)


# 划分的数据包含电压和容量
def dataset_divide(data, divide_rate, time_step):
    sc = MinMaxScaler()
    sc_x = sc.fit_transform(data[:, 1:])
    sc_y = sc.fit_transform(np.expand_dims(data[:, 0], axis=1))
    data = np.concatenate((sc_y, sc_x), axis=1)
    x, y = sliding_windows(data, time_step)
    y = y[:,0]
    train_size = int(len(y) * divide_rate)

    data_x = Variable(torch.Tensor(np.array(x)))
    data_y = Variable(torch.Tensor(np.array(y)))

    train_x = Variable(torch.Tensor(np.array(x[0:train_size])))
    train_y = Variable(torch.Tensor(np.array(y[0:train_size])))

    test_x = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    test_y = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

    return data_x, data_y, train_x, train_y, test_x, test_y, sc


# 这里重写个划分方法是因为上面的方面得到的数据都是三维的[samples,time steps,feature],但GPR需要的输入是(n_samples_X, n_features)
def divideData(data, divide_rate, time_step):
    sc = MinMaxScaler()
    data_set = sc.fit_transform(data)

    x, y = sliding_windows(data_set, time_step)
    x = x.reshape(x.shape[0],x.shape[1])
    train_size = int(len(y) * divide_rate)

    data_x = Variable(torch.Tensor(np.array(x)))
    data_y = Variable(torch.Tensor(np.array(y)))

    train_x = Variable(torch.Tensor(np.array(x[0:train_size])))
    train_y = Variable(torch.Tensor(np.array(y[0:train_size])))

    test_x = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    test_y = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    return data_x, data_y, train_x, train_y, test_x, test_y, sc


def prepare_new_tensor(data, predicted_value):
    window_size = data.shape[1]
    new_tensor = data.clone().detach()
    new_tensor[0, 0:window_size-1, 0] = data[0, 1:window_size+1, 0]
    new_tensor[0, window_size-1, 0] = predicted_value
    return new_tensor


def prepare_dataset(data, divide_rate, time_step, sc=None):
    if sc is None:
        sc = MinMaxScaler()
        data_set = sc.fit_transform(data)
    elif sc == 'off':
        data_set = data
    else:
        data_set = sc.fit_transform(data)

    x, y = sliding_windows(data_set, time_step)
    train_size = int(len(y) * divide_rate)

    data_x = Variable(torch.Tensor(np.array(x)))
    data_y = Variable(torch.Tensor(np.array(y)))

    train_x = Variable(torch.Tensor(np.array(x[0:train_size])))
    train_y = Variable(torch.Tensor(np.array(y[0:train_size])))

    test_x = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    test_y = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

    return data_x, data_y, train_x, train_y, test_x, test_y, sc


def fft_filter(x, n_components, to_real=True):
    n = len(x)
    fft = np.fft.fft(x, n)
    PSD = fft * np.conj(fft) / n
    _mask = PSD > n_components
    fft = _mask * fft

    clean_data = np.fft.ifft(fft)

    if to_real:
        clean_data = clean_data.real

    return clean_data


def get_segments(data, segment_width):
    _length = data.shape[1]
    Index = np.arange(0, _length, segment_width)
    segments = []
    for i in range(len(Index) - 1):
        segments.append(data[:, Index[i]:Index[i + 1]])
    return segments


def num2tensor(*args):
    return (torch.from_numpy(value).float() for value in args)


def tensor2num(*args):
    return (value.detach().numpy() for value in args)


def scale(*args, feature_range=(0, 1)):
    scalers = tuple(MinMaxScaler(feature_range=feature_range).fit(arg) for arg in args)
    temp = (scaler.transform(arg) for arg, scaler in zip(args, scalers))
    return temp, scalers


def concatenate_data_set(data_set):
    if len(data_set) == 1:
        return data_set[0]

    if len(data_set) == 0:
        raise ValueError

    unified_data_set = _unify_dimension(data_set)
    data = unified_data_set[0]

    for i in range(len(data_set) - 1):
        data = np.concatenate((data, unified_data_set[i+1]), axis=0)

    return data


def _unify_dimension(data_set):
    data = data_set[0]
    dimension = data.shape[1]

    for i in range(len(data_set)-1):
        data = data_set[i+1]
        if data.shape[1]<dimension:
            dimension = data.shape[1]
    for i in range(len(data_set)):
        data_set[i] = data_set[i][:, 0:dimension]

    return data_set


def read_matrix_from_file(filename: str):
    row = []
    if Path(filename).is_file():
        with open(filename) as file:
            for line in file:
                col = [float(x) for x in line.split(',')]
                row.append(col)
    # 检查 row 维度是否一致
    lens_ = row[0]
    for col in row:
        if lens_ != len(col):
            return row
    return np.array(row)


def write_matrix_to_file(filename: str, matrix: list):
    """
    :param filename: filename and path
    :param matrix:
    :return: None
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    # An error is reported when shape = (n,), so the save logic is handled
    temp = np.asarray(matrix)
    if temp.ndim == 1:
        matrix = np.asarray(matrix)
        np.savetxt(filename, matrix, fmt='%lf', delimiter=',')
        return

    with open(filename, 'w') as of:
        for row in matrix:
            for index, column in enumerate(row):
                if index == len(row)-1:
                    of.write(str(column))
                else:
                    of.write(str(column)+",")
            of.write("\n")


def get_result_path(results_type, results_name, file_extension, cfg, is_temp=True):
    """
    :param results_type: result dir type - generation | transfer
    :param results_name: filename
    :param file_extension: file extension name - csv, txt
    :param cfg: cfg file, directory
    :param is_temp: save file in result_ or results - means temp results or final results
    :return: absolute path
    """
    base_file_name = results_type + '/' + results_name + '.' + file_extension
    path = cfg.root_path_str + '/' + cfg.result_path
    if is_temp:
        path = path + '_/' + base_file_name
    else:
        path = path + '/' + base_file_name
    return path


def get_category_result_path(model_name, train_battery, test_battery, category, cfg):
    base_file_name = 'Category_' + category + '_from_' + train_battery + '/' + test_battery + '_' + \
                     model_name + '_from_' + train_battery + '.csv'

    path = cfg.root_path_str + cfg.result_path + base_file_name
    return path


def get_data(battery_name, data_path):
    voltages = pd.read_csv(data_path + battery_name + '/voltage.csv', header=None).values
    currents = pd.read_csv(data_path + battery_name + '/current.csv', header=None).values
    capacities = pd.read_csv(data_path + battery_name + '/capacity.csv', header=None).values
    return voltages, currents, capacities


def ask_for_overwrite(path, results_name='', check_file=True):
    if check_file:
        if os.path.exists(path):
            message = "Experiment result for " + results_name + " exists, do you want to overwrite it? [y/n]: "
            action = input(message)
            if action.lower() == 'y' or action.lower() == 'yes':
                return True
            else:
                return False
        else:
            return True