#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hype.graph import eval_reconstruction, load_adjacency_matrix
import argparse
import numpy as np
import torch
import os
import timeit
from hype import MANIFOLDS, MODELS
import pandas

def saveCoordinates(object_list, coordinates, filename):
    df = pandas.DataFrame(coordinates)
    df.loc[:, 'object'] = pandas.Series(object_list, index=df.index)

    cols = df.columns.tolist()
    cols_modified = cols[-1:] + cols[:-1]


    df = df[cols_modified]
    df.to_csv(filename, index=False)

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('file', help='Path to checkpoint')
parser.add_argument('-workers', default=1, type=int, help='Number of workers')
parser.add_argument('-sample', type=int, help='Sample size')
parser.add_argument('-quiet', action='store_true', default=False)
args = parser.parse_args()


chkpnt = torch.load(args.file)
dset = chkpnt['conf']['dset']
if not os.path.exists(dset):
    raise ValueError("Can't find dset!")

format = 'hdf5' if dset.endswith('.h5') else 'csv'
dset = load_adjacency_matrix(dset, format, objects=chkpnt['objects'])

sample_size = args.sample or len(dset['ids'])
sample = np.random.choice(len(dset['ids']), size=sample_size, replace=False)

adj = {}

for i in sample:
    end = dset['offsets'][i + 1] if i + 1 < len(dset['offsets']) \
        else len(dset['neighbors'])
    adj[dset['ids'][i]] = set(dset['neighbors'][dset['offsets'][i]:end])
manifold = MANIFOLDS[chkpnt['conf']['manifold']]()

manifold = MANIFOLDS[chkpnt['conf']['manifold']]()
model = MODELS[chkpnt['conf']['model']](
    manifold,
    dim=chkpnt['conf']['dim'],
    size=chkpnt['embeddings'].size(0),
    sparse=chkpnt['conf']['sparse']
)
model.load_state_dict(chkpnt['model'])

lt = chkpnt['embeddings']
if not isinstance(lt, torch.Tensor):
    lt = torch.from_numpy(lt).cuda()

tstart = timeit.default_timer()
meanrank, maprank = eval_reconstruction(adj, model, workers=args.workers,
    progress=not args.quiet)
etime = timeit.default_timer() - tstart

print(f'Mean rank: {meanrank}, mAP rank: {maprank}, time: {etime}')


# sp_filename = args.file.split('.', 1)
sp_filename = ""
temp_filename = args.file
while sp_filename == "":
    sp_filename, temp_filename = temp_filename.split('.', 1)
csv_filename = './' + sp_filename + '_' + chkpnt['conf']['manifold'] + str(chkpnt['conf']['dim']) + 'D_emb.csv'
lt = torch.Tensor.cpu(lt)
saveCoordinates(dset['objects']['obj'], lt.numpy(), csv_filename)
