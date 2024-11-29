import argparse
import pandas as pd
from model.CAEModel import CNNAEFeature
from config import cfg_from_yaml_file, cfg
import numpy as np
import torch
from utils.mybox import read_config, myplot, parse_MIT, find_bolder, reshape_c, mapping_c
from utils.data_processing import write_matrix_to_file, get_result_path
from torch.autograd import Variable
from scipy import signal
from sklearn import metrics
import itertools


class GenerateCase():
    def __init__(self,):

        self.args = self._read_config()
        self.num_epochs = self.args.CAE.num_epochs
        self.batch_size = self.args.CAE.batch_size
        self.learning_rate = self.args.CAE.learning_rate
        self.every_epoch_print = self.args.CAE.every_epoch_print
        self.feature_num = self.args.CAE.feature_num  
        self.device = self.args.CAE.device
        self.gen_num = self.args.CAE.gen_num        
        self.expand = 20     
        self.cut = 0.8
        self.savepath = './net/cae.pth'

    def _read_config(self):
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--model_file', type=str, default='cfgs/MIT_configs/model_config.yaml')
        args = parser.parse_args()
        cfg_from_yaml_file(args.model_file, cfg)
        return cfg

    def generator(self, matrix, loading=False, is_temp=True):
        """
        :param matrix: [border lower, cluster closet, border upper] capacity array
        :param loading: loading saved model Boolean value, default False
        :param is_temp: savepath, Ture will saved in 'results_', False will saved in 'results'
        :return: generated curves
        """

        torch.manual_seed(1)  
        torch.cuda.manual_seed_all(1)  

        lower_c = matrix[0]
        upper_c = matrix[2]
        closet_c = matrix[1]
        maxs = len(upper_c)

        lower_insert = reshape_c(lower_c, maxs)  
        closet_insert = reshape_c(closet_c, maxs) 

        capacities = []
        for i in range(self.expand):
            capacities.append(lower_insert)
            capacities.append(upper_c)
            capacities.append(closet_insert)
        capacities = np.asarray(capacities)

        capacities = np.expand_dims(capacities, axis=1)  
        # capacities = capacities.swapaxes(1, 2)
        # print(capacities.shape)
        traindata = Variable(torch.tensor(capacities)).float()

        if not loading:
            model = CNNAEFeature(capacities.shape[1], capacities.shape[2], self.feature_num, self.num_epochs, self.batch_size,
                                 self.learning_rate, self.every_epoch_print, self.device)
            model.fit(traindata)  # train process
        else:
            model = torch.load(self.savepath)

        output = model.get_decode_capacity(traindata).detach().numpy()
        print_output = output.squeeze()
        count = 0
        count += metrics.r2_score(print_output[0], lower_insert)
        count += metrics.r2_score(print_output[1], upper_c)
        count += metrics.r2_score(print_output[2], closet_insert)
        print('Train ACC Rate: ', count/3)


        feature = model.get_feature(traindata)
        feature_array = feature.detach().numpy().squeeze()
        feature_array = np.asarray(feature_array)


        _indices = []
        nums = [i for i in range(0, 3)]

        for num in itertools.combinations(nums, 2):
            _indices.append(num)
        new_features = []
        for j in range(len(_indices)):
            temp = []
            for i in range(self.feature_num):
                min_ = min(feature_array[_indices[j][0]][i], feature_array[_indices[j][1]][i])
                max_ = max(feature_array[_indices[j][0]][i], feature_array[_indices[j][1]][i])
                temp.append(np.linspace(min_*0.65, max_*1.24, self.gen_num))
            new_features.append(temp)
        new_features = np.asarray(new_features)
        new_features = new_features.swapaxes(0, 2)
        new_features = new_features.swapaxes(1, 2)
        new_features = new_features.reshape((int(new_features.shape[0]*new_features.shape[1]), 75))
        new_features = np.expand_dims(new_features, axis=1)
        new_features = Variable(torch.tensor(new_features)).float()
        new_curve = model.get_decoder(new_features).detach().numpy().squeeze()


        outputs = []
        for i in range(new_curve.shape[0]):
            curve = signal.savgol_filter(new_curve[i], 31, 1)
            outputs.append(curve.tolist())

        if not loading:
            torch.save(model, self.savepath)

        if is_temp:
            path = get_result_path(args.genepath, 'geneations', 'csv', cfg)
            write_matrix_to_file(path, outputs)
            path = get_result_path(args.genepath, 'refers', 'csv', cfg)
            write_matrix_to_file(path, matrix)
        elif not is_temp:
            path = get_result_path(args.genepath, 'geneations', 'csv', cfg, is_temp=is_temp)
            write_matrix_to_file(path, outputs)
            path = get_result_path(args.genepath, 'refers', 'csv', cfg, is_temp=is_temp)
            write_matrix_to_file(path, matrix)
        return outputs


if __name__ == '__main__':
    args = read_config('cfgs/MIT_configs/base_config.yaml')
    Gen = GenerateCase()
    cluster_ = 0
    indics, matrix = find_bolder(cluster_)

    cells, indexs = parse_MIT(cluster_)
    mins = matrix[0].shape[0]
    maxs = matrix[2].shape[0]

    gen_outputs = Gen.generator(matrix, loading=True)  # load and eval
    map_outputs = mapping_c(gen_outputs, mins, maxs)



