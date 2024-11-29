from classification.classification import classify
from generation.generator import GenerateCase
from selection.selector import selection
from transfer.transfer import Transfer
from utils.mybox import read_config, mapping_c, find_bolder, parse_MIT
from utils.draw import show_prediction
from config import cfg
import numpy as np
from utils.data_processing import prepare_dataset, write_matrix_to_file, get_result_path


if __name__ == '__main__':
    cells, indics = parse_MIT(0)
    for cell in cells:
        input = cell

        cluster, true_cluster = classify(input, cycle=100)
        indics, matrix = find_bolder(cluster)
        mins = matrix[0].shape[0]
        maxs = matrix[2].shape[0]
        path = get_result_path(args.genepath, 'geneations', 'csv', cfg)
        gen_outputs = np.loadtxt(path, dtype=float, delimiter=',')
        transfer = Transfer()
        source, target, endsline = selection(input, gen_outputs, cluster)
        prediction = transfer.predict(source, target, endsline, input, is_plot=True)  # load and eval
        show_prediction(input)

