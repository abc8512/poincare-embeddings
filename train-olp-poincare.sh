#!/bin/sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python3 embed.py \
       -dim 4 \
       -lr 0.1 \
       -epochs 1000 \
       -negs 50 \
       -burnin 20 \
       -ndproc 4 \
       -model distance \
       -manifold lorentz \
       -dset data/olp_goals_connected.csv \
       -checkpoint data/olp_goals_connected.pth \
       -batchsize 10 \
       -eval_each 1 \
       -fresh \
       -sparse \
       -train_threads 1
