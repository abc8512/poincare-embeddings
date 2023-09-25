import matplotlib.pyplot as plt
import pandas
import numpy as np
from hype import MANIFOLDS, MODELS
import torch as th
from enum import Enum
import os
from hype.graph import eval_reconstruction, load_adjacency_matrix
import timeit


class HypType(Enum):
     Lorentz = 'lorentz'
     Poincare = 'poincare'


class Reconstruct:
    def __init__(self) -> None:
        self.embedding_dict = {}
        self.label_dict = {}
        self.graph = None
        self.embedding_filename = None
        pass


    @classmethod
    def fromFiles(cls, embedding_filename, graph_filename):
        """_summary_

        Args:
            embedding_filename (str): a file with embedded coordinates. 'csv' file.
            graph_filename (_type_): graph data filename

        Returns:
            _type_: _description_
        """
        c = cls()
        c.load_embedding(embedding_filename)
        c.load_graph(graph_filename)

        return c
    

    def load_embedding(self,
                       filename: str, delim: str=',', ) -> dict:
        self.data = pandas.read_table(filename, delimiter=delim)
        self.data.rename(columns={self.data.columns[0]:'node'}, inplace=True)

        temp_coord = np.array(self.data.iloc[:, 1:])
        self.node_names = self.data['node'].values
        for i in range(self.data.shape[0]):
            self.embedding_dict[self.node_names[i]] = temp_coord[i]

        if HypType.Lorentz.value in filename:
            self.manifold_type = HypType.Lorentz
            self.dim = temp_coord.shape[1] - 1
        elif HypType.Poincare.value in filename:
            self.manifold_type = HypType.Poincare
            self.dim = temp_coord.shape[1]
        else:
            ValueError("Invalid Hyperbolic Space Type.")
        self.manifold = MANIFOLDS[self.manifold_type.value]()

        return self.embedding_dict
    
    
    def load_graph(self,
                   filename: str, delim: str=',') -> None:
        self.graph = pandas.read_table(filename, delimiter=delim)


    def load_files(self, embedding_filename, graph_filename) -> None:
        self.load_embedding(embedding_filename)
        self.load_graph(graph_filename)
    

    def convert_manifold(self, manifold_type: HypType) -> None:
        if manifold_type == HypType.Lorentz:
            if self.manifold_type == HypType.Poincare:
                self._poincare_to_lorentz()
                self.manifold_type = manifold_type
            else:
                pass
        elif manifold_type == HypType.Poincare:
            if self.manifold_type == HypType.Lorentz:
                self._lorentz_to_poincare()
                self.manifold_type = manifold_type
            else:
                pass

        self.manifold = MANIFOLDS[self.manifold_type.value]()


    def _lorentz_to_poincare(self) -> None:
        if self.manifold_type == HypType.Lorentz:
            x = np.array(list(self.embedding_dict.values()))
            print(x.shape)
            y = x[:, 1:] / (x[:, 0, np.newaxis] + 1)
            for i in range(self.data.shape[0]):
                self.embedding_dict.update({self.node_names[i]: y[i, :]})


    def _poincare_to_lorentz(self) -> None:
        if self.manifold_type == HypType.Poincare:
            for i in range(self.data.shape[0]):
                x = self.embedding_dict[self.node_names[i]]
                norm_x = np.linalg.norm(x)
                self.embedding_dict.update({self.node_names[i]: np.append(1 + norm_x**2, 2*x) / (1 - norm_x**2)})


    def sort_by_distance(self) -> None:
        self._calculate_distance_from_origin()
        sorted_df = self.data.sort_values(by='distance')
        return sorted_df


    def _calculate_distance_from_origin(self) -> None:
        if not 'distance' in self.data.columns:
            origin = np.zeros(self.dim)
            if self.manifold_type == HypType.Lorentz:
                origin = np.append(1, origin)
            dist_origin = self.manifold.distance(th.Tensor(origin), 
                                                    th.Tensor(np.array(list(self.embedding_dict.values())))).numpy()
            df = pandas.DataFrame({'id': list(self.node_names), 'distance': dist_origin})
            self.data.insert(len(self.data.columns), 'distance', dist_origin)
        

    def _save_coordinates(self, object_list, coordinates, filename) -> None:
        df = pandas.DataFrame(coordinates)
        df.loc[:, 'object'] = pandas.Series(object_list, index=df.index)

        cols = df.columns.tolist()
        cols_modified = cols[-1:] + cols[:-1]


        df = df[cols_modified]
        df.to_csv(filename, index=False)


    def reconstruction(self, filename: str, workers: int=1, num_sample: int=None, quiet: bool=False) -> None:
        """_summary_
        copy from reconstruction.py

        Args:
            filename (str): {}.pth.best file
            workers (int, optional): _description_. Defaults to 1.
            num_sample (int, optional): _description_. Defaults to None.
            quiet (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
        """
        np.random.seed(42)
        chkpnt = th.load(filename)
        dset = chkpnt['conf']['dset']
        if not os.path.exists(dset):
            raise ValueError("Can't find dset!")
        
        format = 'hdf5' if dset.endswith('.h5') else 'csv'
        dset = load_adjacency_matrix(dset, format, objects=chkpnt['objects'])

        sample_size = num_sample or len(dset['ids'])
        sample = np.random.choice(len(dset['ids']), size=sample_size, replace=False)

        adj = {}

        for i in sample:
            end = dset['offsets'][i + 1] if i + 1 < len(dset['offsets']) \
                else len(dset['neighbors'])
            adj[dset['ids'][i]] = set(dset['neighbors'][dset['offsets'][i]:end])
        
        manifold = MANIFOLDS[chkpnt['conf']['manifold']]()
        model = MODELS[chkpnt['conf']['model']](
            manifold,
            dim=chkpnt['conf']['dim'],
            size=chkpnt['embeddings'].size(0),
            sparse=chkpnt['conf']['sparse']
        )
        model.load_state_dict(chkpnt['model'])
        lt = chkpnt['embeddings']
        if not isinstance(lt, th.Tensor):
            lt = th.from_numpy(lt).cuda()

        tstart = timeit.default_timer()
        meanrank, maprank = eval_reconstruction(adj, model, workers=workers, progress=not quiet)
        etime = timeit.default_timer() - tstart

        print(f'Mean rank: {meanrank}, mAP rank: {maprank}, time: {etime}')
        print('Manifold: {}, dim: {}'.format(chkpnt['conf']['manifold'], chkpnt['conf']['dim']))

        sp_filename = ""
        temp_filename = filename
        while sp_filename == "":
            sp_filename, temp_filename = temp_filename.split('.', 1)
        csv_filename = '.' + sp_filename + '_' + chkpnt['conf']['manifold'] + str(chkpnt['conf']['dim']) + 'D_emb.csv'
        lt = th.Tensor.cpu(lt)
        self._save_coordinates(dset['objects']['obj'], lt.numpy(), csv_filename)
        
        self.embedding_filename = csv_filename





    # def plot_embedding2D(self, )





