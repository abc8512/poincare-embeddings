import matplotlib.pyplot as plt
import pandas
import numpy as np
from hype import MANIFOLDS
import torch as th


def load_embedding(filename: str, delim=',') -> dict:
    emb_data = pandas.read_table(filename, delimiter=delim)
    emb_data.rename(columns={emb_data.columns[0]:'node'}, inplace=True)
    # emb_data.columns = ['node', 'x', 'y']

    # if emb_data.dtypes['node'] != np.number:
    #     try:
    #         emb_data = emb_data.loc[(emb_data.node.apply(lambda x: x not in ['u', 'v'])), :]     # what does it mean?
    #         # emb_data['node'] = emb_data.node.astype('int')
    #         emb_data = emb_data.sort_values(by='node').reset_index(drop=True)     # why does it sort?
    #     except ValueError as e:
    #         pass

    temp_coord = np.array(emb_data.iloc[:, 1:]) 
    if 'lorentz' in filename:
        temp_coord = lorentzToPoincare(temp_coord)
    emb_coord = np.array(temp_coord[:, :2])

    embedding_dict = {}
    node_attr = emb_data['node'].values
    for i in range(emb_data.shape[0]):
        embedding_dict[node_attr[i]] = emb_coord[i]
    
    return embedding_dict


def plot_embedding2D(embedding_dict: dict, label_dict: dict, edge_list,
                     label_frac: float=0.001, plot_frac: float=0.6, title: str=None, save_fig: bool=False, save_filename: str=None, 
                     plot_edge: bool=False, groups: list=[]) -> None:
    # colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k', 'w']
    fig = plt.figure(figsize=(9, 9))
    plt.grid('off')
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.axis('off')
    if title != None:
        plt.title(title, size=16)
    ax = plt.gca()
    colors = plt.get_cmap('Paired')

    embed_vals = np.array(list(embedding_dict.values()))
    keys = list(embedding_dict.keys())  # node names

    min_dist_2 = label_frac * max(embed_vals.max(axis=0) - embed_vals.min(axis=0)) ** 2
    labeled_vals = np.array([2*embed_vals.max(axis=0)])

    # embed_vals의 distance가 가장 작은 값들 --> higher hierarchy
    manifold = MANIFOLDS['poincare']()
    groups = groups + [keys[i] for i in np.argsort(manifold.distance(th.Tensor(np.array([0, 0])), th.Tensor(embed_vals)).numpy())][:10]

    # groups.insert(0, 'mammal.n.01')     # mammal 이 가장 높은 hierarchy 이길 바라지만, 학습이 충분히 안되면 embedding 결과는 조금 다름.
    manifold_Eu = MANIFOLDS['euclidean']()

    # plot some higher hierarchy first.
    for key in groups:
        # skip drawing if the point is too close to the already plotted points.
        # if np.min(manifold_Eu.distance(th.Tensor(embedding_dict[key]), th.Tensor(labeled_vals)).numpy()) < min_dist_2:
        if False:
            continue
        else:
            label_splitted = key.split('.')
            label_str = label_splitted[0]
            if len(label_splitted) > 1:
                if '_' in label_splitted[1]:
                    label_str = label_str + '.s' + label_splitted[1].split('.')[0].split('_')[1]
            _ = ax.scatter(embedding_dict[key][0], embedding_dict[key][1], s=40, color=colors(label_dict[key]))
            props = dict(boxstyle='round', lw=2, edgecolor='black', alpha=0.5, facecolor='wheat')
            _ = ax.text(embedding_dict[key][0], embedding_dict[key][1]+0.01, s=label_str, 
                            fontsize=12, verticalalignment='top', bbox=props)
            labeled_vals = np.vstack((labeled_vals, embedding_dict[key]))

    # plot lower hierarchy points
    n = int(plot_frac*len(embed_vals))  # plot 'n' randomly chosen data
    for i in np.random.permutation(len(embed_vals))[:n]:
        # plot data points
        _ = ax.scatter(embed_vals[i][0], embed_vals[i][1], s=40, color=colors(label_dict[keys[i]]))
        # choose whether drawing a label or not.
        if np.min(manifold_Eu.distance(th.Tensor(embed_vals[i]), th.Tensor(labeled_vals)).numpy()) < min_dist_2:
            continue
        else:
            props = dict(boxstyle='round', lw=2, edgecolor='black', alpha=0.5)
            _ = ax.text(embed_vals[i][0], embed_vals[i][1]+0.02, s=keys[i].split('.')[0],
                        fontsize=11, verticalalignment='top', bbox=props)
            labeled_vals = np.vstack((labeled_vals, embed_vals[i]))

    if plot_edge:
        for i in range(edge_list.shape[0]):
            e_x1 = embedding_dict[edge_list['id1'][i]]
            e_x2 = embedding_dict[edge_list['id2'][i]]
            _ = ax.plot([e_x1[0], e_x2[0]], [e_x1[1], e_x2[1]], '-', c='black', linewidth=0.1, alpha=0.3)

    
    if save_fig:
        plt.savefig(save_filename)
    plt.show()

def calc_distances_from_zero(filename, delim: str=',') -> pandas.DataFrame:
    emb_data = pandas.read_table(filename, delimiter=delim)
    emb_data.rename(columns={emb_data.columns[0]:'node'}, inplace=True)

    temp_coord = np.array(emb_data.iloc[:, 1:]) 
    if 'lorentz' in filename:
        temp_coord = lorentzToPoincare(temp_coord)
    emb_coord = np.array(temp_coord)

    embedding_dict = {}
    node_attr = emb_data['node'].values
    for i in range(emb_data.shape[0]):
        embedding_dict[node_attr[i]] = emb_coord[i]

    l_manifold = MANIFOLDS['poincare']()

    node_keys = list(embedding_dict.keys())
    dist_origin = l_manifold.distance(th.Tensor(np.zeros(emb_coord.shape[1])), th.Tensor(np.array(list(embedding_dict.values())))).numpy()

    df = pandas.DataFrame({'id': node_keys, 'distance': dist_origin})

    return df


def lorentzToPoincare(x):
    return x[:, 1:] / (x[:, 0, np.newaxis] + 1)




