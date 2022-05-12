#!/usr/bin/env python

import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", dest='dataset_csv', help='CSV file with dataset')
parser.add_argument("-k", dest='kfolds', help='precomputed kfolds pickle file to use')

args = parser.parse_args()

cvtag = args.kfolds.split('_')[-1].split('.')[0]

# load the dataset
df = pd.read_csv(args.dataset_csv)

# load the kfold splits
with open(args.kfolds, 'rb') as f:
    kfolds = pickle.load(f)

for k, kfold in enumerate(kfolds):
    print(f'fold {k}')

    train_idx = kfold['train_idx']
    val_idx = kfold['val_idx']
    test_idx = kfold['test_idx']

    df_train = df.loc[train_idx, :]
    df_val = df.loc[val_idx, :]
    df_test = df.loc[test_idx, :]

    df_train.to_csv(f'input_train_{k}.csv', index=False)
    df_val.to_csv(f'input_val_{k}.csv', index=False)
    df_test.to_csv(f'input_test_{k}.csv', index=False)

