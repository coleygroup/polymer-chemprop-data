#!/usr/bin/env python

import pandas as pd
import numpy as np
import pickle
import argparse

from sklearn.ensemble import RandomForestRegressor

from functools import partial
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

parser = argparse.ArgumentParser()
parser.add_argument("-f", dest='dataset_pkl', help='Pickle file with dataset X and Y')
parser.add_argument("-k", dest='kfolds', help='Pickle file with precomputed kfoldsto use')
parser.add_argument("--train_on_val", dest='train_on_val', help='whether to merge the validation with the training set', default=False, action='store_true')
parser.add_argument("--hopt", dest='hopt', help='number of iterations of hyperparam optimization', default=0, type=int)
args = parser.parse_args()

if args.train_on_val is True and args.hopt > 0:
    raise ValueError('either use hyperparam opt or merge train+val sets')

# load the dataset
with open(args.dataset_pkl, 'rb') as f:
    data = pickle.load(f)
    X = data['X']
    Y = data['Y']
    # these are from:
    # X = pd.DataFrame(fps, columns=[f'bit-{x}' for x in range(nBits)])
    # Y = df.loc[:, ['EA vs SHE (eV)', 'IP vs SHE (eV)']]

# load the kfold splits
with open(args.kfolds, 'rb') as f:
    kfolds = pickle.load(f)


# define Hyperopt objective function to be minimized
def hopt_objective(params, X_train, Y_train, X_val, Y_val):

    # extract params
    n_estimators = int(params['n_estimators'])
    max_depth = int(params['max_depth'])
    min_samples_leaf = int(params['min_samples_leaf'])
    min_samples_split = int(params['min_samples_split'])

    # instantiate model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                  min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                  random_state=42, n_jobs=12)
    # fit
    model.fit(X_train, Y_train)
    # predict
    Y_pred = model.predict(X_val)
    # evaluate
    rmse = np.sqrt(np.mean((np.array(Y_pred) - np.array(Y_val))**2))
    return {'loss': rmse, 'status': STATUS_OK}

# Hyperopt optimization
def hopt_optimize(trial, X_train, Y_train, X_val, Y_val, max_evals=10):
    hspace={'n_estimators': hp.quniform('n_estimators', 100, 500, 1),
            'max_depth': hp.quniform('max_depth', 5, 20, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 6, 1)}

    fmin_objective = partial(hopt_objective, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val)
    best=fmin(fn=fmin_objective, space=hspace, algo=tpe.suggest, trials=trial, max_evals=max_evals, rstate=np.random.RandomState(42))
    return best


# ================
# Cross-Validation
# ================

for k, kfold in enumerate(kfolds):
    print()
    print(f'fold {k}')

    train_idx = kfold['train_idx']
    val_idx = kfold['val_idx']
    test_idx = kfold['test_idx']

    if args.train_on_val is True:
        exp_num = len(train_idx) + len(val_idx)
        train_idx = np.concatenate((train_idx, val_idx))
        if len(train_idx) != exp_num:
            raise ValueError(f"expected {exp_num}, found {len(train_idx)}")

    # get input features
    X_train = X.loc[train_idx, :]
    X_val = X.loc[val_idx, :]
    X_test = X.loc[test_idx, :]

    # get labels
    Y_train = Y.loc[train_idx, :]
    Y_val = Y.loc[val_idx, :]
    Y_test = Y.loc[test_idx, :]

    # save target labels for analysis
    Y_test.to_csv(f'input_test_{k}.csv', index=False)

    # define model
    if args.hopt > 0:
        # do hyperparam optimization on validation set
        trial=Trials()
        best = hopt_optimize(trial, X_train, Y_train, X_val, Y_val, args.hopt)

        # extract params
        n_estimators = int(best['n_estimators'])
        max_depth = int(best['max_depth'])
        min_samples_leaf = int(best['min_samples_leaf'])
        min_samples_split = int(best['min_samples_split'])

        # instantiate model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                      random_state=42, n_jobs=12)

        # merge train and val sets
        exp_num = len(train_idx) + len(val_idx)
        train_and_val_idx = np.concatenate((train_idx, val_idx))
        if len(train_and_val_idx) != exp_num:
            raise ValueError(f"expected {exp_num}, found {len(train_and_val_idx)}")
        
        X_train = X.loc[train_and_val_idx, :]
        Y_train = Y.loc[train_and_val_idx, :]
        
    else:
        # default
        model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=12)
    
    # train model
    model.fit(X_train, Y_train)

    # make predictions
    Y_pred = model.predict(X_test)

    df_predictions = pd.DataFrame(np.array(Y_pred), columns=Y.columns)
    df_predictions.to_csv(f'predictions_{k}.csv', index=False)

