from experiment.generation.generator import GenerateCase
import pandas as pd
from config import cfg_from_yaml_file, cfg
from utils.mybox import read_config, myplot, parse_MIT, find_bolder, reshape_c, mapping_c
from utils.data_processing import write_matrix_to_file, read_matrix_from_file, get_result_path
import numpy as np
from sklearn import metrics
from classification.classification import classify
from selection.selector import selection


def show_selection(args):

    path = get_result_path(args.selepath, 'source', 'csv', cfg)
    source = np.loadtxt(path, dtype=float, delimiter=',')
    path = get_result_path(args.selepath, 'target', 'csv', cfg)
    target = read_matrix_from_file(path)
    path = get_result_path(args.selepath, 'endsline', 'csv', cfg)
    endsline = read_matrix_from_file(path)

    plt = myplot(xlabel='Cycle', ylabel='Capacity', title='Selection Result')
    plt.plot(source, c='blue', ls='-.', lw=0.8, marker='*', label='source', markevery=20)
    plt.plot(target, c='pink', ls='-.', lw=0.8, marker='*', label='target', markevery=20)
    plt.plot(endsline, c='green', ls='--', label='endsline')
    plt.legend(loc="best")
    plt.show()


def synthetic_experiment(args):
    input = 'cell40'
    cluster, true_cluster = classify(input, cycle=100)
    path = get_result_path(args.genepath, 'geneations', 'csv', cfg)
    gen_outputs = np.loadtxt(path, dtype=float, delimiter=',')
    source, target, endsline = selection(input, gen_outputs=gen_outputs, cluster_=cluster)


if __name__ == '__main__':
    config_file = 'cfgs/MIT_configs/base_config.yaml'
    args = read_config(config_file)
    # synthetic_experiment(args)
    show_selection(args)
