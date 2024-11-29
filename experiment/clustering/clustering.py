from sklearn.cluster import KMeans
from utils.mybox import read_config, myplot, parse_MIT
import numpy as np
import pandas as pd


def do_cluster():
    args = read_config('cfgs/MIT_configs/base_config.yaml')

    Q = []
    C = []
    cells = []
    # for i in range(10):
    for i in range(150):
        try:
            q = pd.read_csv(args.data_path + 'cell' + str(i + 1) + "\\qdlin.csv", header=None).values
            c = pd.read_csv(args.data_path + 'cell' + str(i + 1) + "\\capacity.csv", header=None).values.flatten()
        except:
            continue
        vars = np.log10(np.var(q[100 - 1] - q[10 - 1]))
        Q.append(vars)
        C.append(c)
        cells.append(i+1)


    k = 3  

    Q = np.asarray(Q).reshape(-1, 1)
    model = KMeans(n_clusters=k).fit(Q)

    labels = model.predict(Q)

    closet = []

    for i in range(model.n_clusters):
        cluster_indices = np.where(labels == i)[0]
        distances = model.transform(Q[cluster_indices])[:, i]
        nearest_point_index = cluster_indices[np.argmin(distances)]
        closet.append([cells[nearest_point_index], i])
        print(f"Cluster {i}: nearest point index = {nearest_point_index}, cellname = {cells[nearest_point_index]}")
    closet = np.asarray(closet)

    result = []
    result.append(cells)
    result.append(labels)
    result = np.asarray(result)
    result = np.swapaxes(result, 0, 1)
    print(labels)
    print(result)
    np.savetxt(args.data_path + 'categories.csv', result, fmt='%d', delimiter=',')
    np.savetxt(args.data_path + 'closet.csv', closet, fmt='%d', delimiter=',')


if __name__ == '__main__':
    do_cluster()

    args = read_config('cfgs/MIT_configs/base_config.yaml')
    result = np.loadtxt(args.data_path + 'categories.csv', dtype=int, delimiter=',')
    plt = myplot(xlabel='x', ylabel='y', title='cluster result')
    tag = 0
    color = ['green', 'gray', 'red']
    for i in range(150):
        try:
            c = pd.read_csv(args.data_path + 'cell' + str(i + 1) + "\\capacity.csv", header=None).values.flatten()
        except:
            continue
        plt.plot(c, c=color[result[tag][1]], lw=1)
        tag += 1
    plt.show()
