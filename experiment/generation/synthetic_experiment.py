from experiment.generation.generator import GenerateCase
import pandas as pd
from config import cfg_from_yaml_file, cfg
from utils.mybox import read_config, myplot, parse_MIT, find_bolder, reshape_c, mapping_c
from utils.data_processing import write_matrix_to_file, read_matrix_from_file, get_result_path
import numpy as np
from sklearn import metrics


def show_generation(args):

    path = get_result_path(args.genepath, 'geneations', 'csv', cfg)
    gen_outputs = np.loadtxt(path, dtype=float, delimiter=',')
    path = get_result_path(args.genepath, 'refers', 'csv', cfg)
    refers = read_matrix_from_file(path)
    mins = len(refers[0])
    maxs = len(refers[2])
    map_outputs = mapping_c(gen_outputs, mins, maxs)

    plt = myplot(xlabel='Cycle', ylabel='Capacity', title='Generation Result')
    cmap = plt.cm.get_cmap('PuBu')
    colors = [cmap(i) for i in np.linspace(0, 1, len(map_outputs))]
    for i in range(len(map_outputs)):
        plt.plot(map_outputs[i], c=colors[int(i % len(map_outputs))], lw=1)
    plt.plot(refers[0], c='red', ls='-.', lw=0.5, marker='*', label='lower', markevery=20)
    plt.plot(refers[1], c='red', ls='--', label='closet')
    plt.plot(refers[2], c='red', ls='-.', lw=0.8, marker='*', label='upper', markevery=20)
    plt.legend(loc="best")
    plt.show()


def show_prediction(args,cellname):

    path = get_result_path(args.predpath, cellname, 'csv', cfg)
    prediction = np.loadtxt(path, dtype=float, delimiter=',')
    capacity = pd.read_csv(args.data_path + cellname + "/capacity.csv", header=None)
    # 计算 R2
    mins_ = min(prediction.shape[0], capacity.shape[0])
    similarity = metrics.r2_score(capacity[0:mins_], prediction[0:mins_])
    plt = myplot(xlabel='Cycle', ylabel='Capacity', title='Prediction Result')
    plt.plot(prediction, c='green', ls='-.', lw=0.8, marker='*', label='prediction', markevery=20)
    plt.plot(capacity, c='red', ls='-.', lw=0.5, label='true')
    plt.text(x=50, 
             y=0.9,  
             s=str("R²: %.2f" % similarity), 
             rotation=1, 
             ha='left',  
             va='baseline', 
             fontdict=dict(fontsize=8, color='black',
                           family='monospace', 
                           weight='light',  
                           ) 
             )
    plt.legend(loc="best")
    plt.show()


def synthetic_experiment(args):
    Gen = GenerateCase()
    cluster_ = 0
    indics, matrix = find_bolder(cluster_)


    cells, indexs = parse_MIT(cluster_)
    mins = matrix[0].shape[0]
    maxs = matrix[2].shape[0]

    # outputs = Gen.generator(lower, upper, closet)      # retrain
    gen_outputs = Gen.generator(matrix, loading=True)  # load and eval
    map_outputs = mapping_c(gen_outputs, mins, maxs)


if __name__ == '__main__':
    config_file = 'cfgs/MIT_configs/base_config.yaml'
    args = read_config(config_file)
    # synthetic_experiment(args)
    show_generation(args)
    show_prediction(args, 'cell40')
