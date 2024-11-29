import pandas as pd
import numpy as np
from config import cfg
from utils.mybox import read_config, parse_MIT, find_bolder, mapping_c, reshape_c
from sklearn import metrics
from utils.data_processing import write_matrix_to_file, get_result_path
from classification.classification import classify


def selection(cellname, gen_outputs, cluster_, is_temp=True):
    """
    :param cellname: selection cellname
    :param gen_outputs: generation outputs
    :param cluster_: cell cluster
    :return: selection curve, cluster closet, Tail data to be concatenated
    """
    args = read_config('cfgs/MIT_configs/base_config.yaml')
    indics, matrix = find_bolder(cluster_)


    cells, indexs = parse_MIT(cluster_)
    mins = matrix[0].shape[0]
    maxs = matrix[2].shape[0]
    map_outputs = mapping_c(gen_outputs, mins, maxs)


    path = get_result_path(args.features, 'delta_qs', 'csv', cfg, is_temp=False)
    delta_qs = np.loadtxt(path, dtype=float, delimiter=',')
    categories = pd.read_csv(args.data_path + "\\categories.csv", header=None).values
    border_deltas = [0, 1, 2]
    cell_feature = 0
    for index in indexs:
        delta_q = delta_qs[index]
        feat_ = np.log10(np.var(delta_q))
        if indics[0] == 'cell' + str(categories[index][0]):
            border_deltas[0] = feat_
        if indics[1] == 'cell' + str(categories[index][0]):
            border_deltas[1] = feat_
        if indics[2] == 'cell' + str(categories[index][0]):
            border_deltas[2] = feat_
        if cellname == 'cell' + str(categories[index][0]):
            cell_feature = feat_


    output_features = np.flip(np.linspace(border_deltas[2], border_deltas[0], len(map_outputs)))
    cycles = []
    for i in range(len(map_outputs)):
        cycles.append(len(map_outputs[i]))
    pccs = np.corrcoef(cycles, output_features)

    curve = 0
    weight = 0.85
    capacity = pd.read_csv(args.data_path + cellname + "\\capacity.csv", header=None).values.flatten()

    results = []
    for i in range(len(map_outputs)):
        temp = []
        r2 = metrics.r2_score(capacity[10:100], map_outputs[i][10:100])
        if r2 < 0:
            r2 = r2 / 300
        capa_score = r2
        feat_score = 1 - np.sqrt(np.square(output_features[i] - cell_feature))
        all_score = (1 - weight) * capa_score + weight * feat_score
        temp.append(i)
        temp.append(all_score)
        temp.append(capa_score)
        temp.append(feat_score)
        results.append(temp)
    results = np.asarray(results)
    r = results[:, 1].astype(float)
    results = results[np.argsort(-r)]
    np.set_printoptions(suppress=True)


    if cell_feature < border_deltas[2] or cell_feature > border_deltas[0]:
        lens_ = 0
        num = 50
        for k in range(num):
            idx = int(results[k][0])
            lens_ += len(map_outputs[idx])
        lens_ = int(lens_ / num)

        curves = []
        for k in range(num):
            idx = int(results[k][0])
            curves.append(reshape_c(map_outputs[idx], lens_))
        curve = np.mean(np.asarray(curves), axis=0)
        mins_ = min(capacity.shape[0], len(curve))
        similarity = metrics.r2_score(capacity[0:mins_], curve[0:mins_])
        print('outside similarity:', capacity.shape[0], len(curve), similarity)
    else:

        lens_ = 0
        num = 50
        for k in range(num):
            idx = int(results[k][0])
            lens_ += len(map_outputs[idx])
        lens_ = int(lens_ / num)

        finals = []
        for k in range(num):
            idx = int(results[k][0])
            curve = reshape_c(map_outputs[idx], lens_)
            finals.append(curve)
        finals = np.asarray(finals)
        curve = np.mean(finals, axis=0)
        mins_ = min(capacity.shape[0], len(curve))
        similarity = metrics.r2_score(capacity[0:mins_], curve[0:mins_])
        print('similarity:', capacity.shape[0], len(curve), similarity)

    source = curve
    target = matrix[1]


    curve = []
    if len(source) > len(target):
        mins_ = len(target)
        cuts = source[mins_:]
        curve.append(cuts)
        curve.append(target[len(target) - len(cuts): len(target)])
        source = source[0:mins_]
        target = target[0:mins_]
    elif len(source) < len(target):
        mins_ = len(source)
        cuts = target[mins_:]
        curve.append(cuts)
        curve.append(source[len(source) - len(cuts): len(source)])
        source = source[0:mins_]
        target = target[0:mins_]
    curve = np.asarray(curve)
    curve = np.mean(curve, axis=0)
    endsline = curve


    if is_temp:
        path = get_result_path(args.selepath, 'source', 'csv', cfg, is_temp=is_temp)
        write_matrix_to_file(path, source)
        path = get_result_path(args.selepath, 'target', 'csv', cfg, is_temp=is_temp)
        write_matrix_to_file(path, target)
        path = get_result_path(args.selepath, 'endsline', 'csv', cfg, is_temp=is_temp)
        write_matrix_to_file(path, endsline)
    elif not is_temp:
        path = get_result_path(args.selepath, 'source', 'csv', cfg, is_temp=is_temp)
        write_matrix_to_file(path, source)
        path = get_result_path(args.selepath, 'target', 'csv', cfg, is_temp=is_temp)
        write_matrix_to_file(path, target)
        path = get_result_path(args.selepath, 'endsline', 'csv', cfg, is_temp=is_temp)
        write_matrix_to_file(path, endsline)
    return source, target, endsline


if __name__ == '__main__':
    args = read_config('cfgs/MIT_configs/base_config.yaml')
    input = 'cell40'
    cluster, true_cluster = classify(input, cycle=100)
    path = get_result_path(args.genepath, 'geneations', 'csv', cfg)
    gen_outputs = np.loadtxt(path, dtype=float, delimiter=',')
    source, target, endsline = selection(input, gen_outputs=gen_outputs, cluster_=cluster)


