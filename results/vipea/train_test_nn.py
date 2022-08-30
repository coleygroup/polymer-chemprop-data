#!/usr/bin/env python

import pandas as pd
import numpy as np
import argparse
import os
import math
import pickle
import shutil
from time import sleep

from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F

from functools import partial
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials, space_eval


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", dest='dataset_pkl', help='Pickle file with dataset X and Y')
parser.add_argument("-k", dest='kfolds', help='Pickle file with precomputed kfolds to use')
parser.add_argument("--hidden_size", dest='hidden_size', type=int, default=128, help='Size of hidden layers')
parser.add_argument("--num_hidden_layers", dest='num_hidden_layers', type=int,  default=2, help='Number of hidden layers (in addition to the input/output layers)')
parser.add_argument("--dropout_rate", dest='dropout_rate', type=float, default=0.5, help='Dropout rate')
parser.add_argument("--batch_size", dest='batch_size', type=int, default=256, help='Batch size')
parser.add_argument("--learning_rate", dest='learning_rate', type=float, default=0.001, help='Learning rate for Adam')
parser.add_argument("--num_epochs", dest='num_epochs', type=int, default=200, help='Max number of epochs allowed')
parser.add_argument("--patience", dest='patience', type=int, default=50, help='Patience (number of epochs) used for early stopping')
parser.add_argument("--checkpoint_folder", dest='checkpoint_folder', default='checkpoints', help='Folder where to temporarily save checkpoints.')
parser.add_argument("--gpu_id", dest='gpu_id', default=0, help='GPU to use if >1 available.')
parser.add_argument("--hopt", dest='hopt', help='number of iterations of hyperparam optimization', default=0, type=int)

args = parser.parse_args()
dropout_rate = args.dropout_rate
num_hidden_layers = args.num_hidden_layers
hidden_size = args.hidden_size
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs
patience = args.patience
checkpoint_folder = args.checkpoint_folder
gpu_id = args.gpu_id
hopt = args.hopt


# =====================
# Classes and functions
# =====================
class FeedForwardMultiOutput(nn.Module):
    def __init__(self, input_size, out_size, hidden_size, num_hidden_layers, dropout_rate):
        super(FeedForwardMultiOutput, self).__init__()
        
        self.input_size = input_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate

        # store layers in a list that will be converted to a torch ModuleList
        fc_stack = []
        ln_stack = []

        # input layer
        fc_in = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer
        # Layer normalization for faster training
        ln_in = nn.LayerNorm(hidden_size)

        fc_stack.append(fc_in)
        ln_stack.append(ln_in)

        for i in range(self.num_hidden_layers):
            fc_i = nn.Linear(hidden_size, hidden_size)
            ln_i = nn.LayerNorm(hidden_size)
            fc_stack.append(fc_i)
            ln_stack.append(ln_i)

        self.fc_stack = nn.ModuleList(fc_stack)
        self.ln_stack = nn.ModuleList(ln_stack)

        # output layer
        self.fc_out = nn.Linear(hidden_size, out_size) # Output layer
        
        # activation function
        self.activation = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        for i in range(self.num_hidden_layers + 1):
            x = self.fc_stack[i](x)
            x = self.ln_stack[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.fc_out(x)
        return x


# Train
def train(device, model, num_epochs, optimizer, loss_function, 
          train_loader, valid_loader, patience, print_interval, checkpoint_folder):

    # Early stopping
    best_valid_loss = np.inf
    es_triggers = 0

    # loss history
    train_losses = []  # to track the training loss as the model trains
    valid_losses = []  # to track the validation loss as the model trains
    best_epoch = 0

    for epoch in range(1, num_epochs+1):
        running_loss = 0
        model.train()

        #for times, data in enumerate(train_loader, 1):
        for data in train_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward and backward propagation
            outputs = model(inputs.view(inputs.shape[0], -1))  # forward pass of the mini-batch
            loss = loss_function(outputs, labels)  # compute the loss
            loss.backward()  # calculate the backward pass
            optimizer.step()  # optimize the weights

            # store losses
            running_loss += loss.item()

        # save train loss
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Early stopping
        valid_loss = validate(model, device, valid_loader, loss_function)
        valid_losses.append(valid_loss)

        # if validation loss is not improving, increase counter
        if valid_loss > best_valid_loss:
            es_triggers += 1
            current_is_best = False
            if es_triggers >= patience:
                break
        # if validation loss is better than best found, save model params
        else:
            # reset counter
            es_triggers = 0
            # save weights
            torch.save(model.state_dict(), checkpoint_folder + '/model.pt')
            # set best loss to current loss
            best_valid_loss = valid_loss
            # current is best so far
            current_is_best = True
            # current epoch is epoch of best model
            best_epoch = epoch

        # Show progress
        if epoch % print_interval == 0:
            epoch_len = len(str(num_epochs))
            print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} | ' +
                         f'valid_loss: {valid_loss:.5f} | ' +
                         f'patience_left: {patience-es_triggers:>{math.ceil(math.log10(patience)+1e-5)}} | ' +
                         (f'*' if current_is_best else ''))
            print(print_msg)

    # load best model from file (i.e. return model with best valid loss, not last one)
    model.load_state_dict(torch.load(checkpoint_folder + '/model.pt'))

    return model, train_losses, valid_losses, best_epoch, best_valid_loss


def validate(model, device, valid_loader, loss_function):
    # Settings
    model.eval()
    running_loss = 0

    # Test validation data
    with torch.no_grad():
        for data in valid_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = model(inputs.view(inputs.shape[0], -1))
            loss = loss_function(outputs, labels)
            running_loss += loss.item()

    return running_loss / len(valid_loader)


# define loss
def loss_function(outputs, labels):
    loss0 = F.mse_loss(outputs[0], labels[0])
    loss1 = F.mse_loss(outputs[1], labels[1])
    loss = torch.mean(loss0**2 + loss1**2)
    return loss


def hopt_objective(params, train_dataset, val_dataset, input_size, out_size, 
                   batch_size, hidden_size, num_epochs, patience, checkpoint_folder, device):
    
    # extract params
    num_hidden_layers = int(params['num_hidden_layers'])
    learning_rate = float(params['learning_rate'])
    dropout_rate = float(params['dropout_rate'])

    # create data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=0)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # init model
    model = FeedForwardMultiOutput(input_size=input_size, 
                                   out_size=out_size, 
                                   hidden_size=hidden_size, 
                                   num_hidden_layers=num_hidden_layers, 
                                   dropout_rate=dropout_rate)
    
    model.to(device) # send model to device

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    model, _, _, _, best_valid_loss = train(device=device, model=model, 
                                            num_epochs=num_epochs, optimizer=optimizer, 
                                            loss_function=loss_function, 
                                            train_loader=train_loader, 
                                            valid_loader=val_loader, 
                                            patience=patience, 
                                            print_interval=1, 
                                            checkpoint_folder=checkpoint_folder)

    return {'loss': best_valid_loss, 'status': STATUS_OK}


# ====
# Main
# ====

# select device 
device = f'cuda:{gpu_id}'

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

# ----------------
# Cross validation
# ----------------
for k, kfold in enumerate(kfolds):
    print()
    print(f'======')
    print(f'Fold {k}')
    print(f'======')

    # create folder for checkpoints
    if os.path.isdir(checkpoint_folder):
        shutil.rmtree(checkpoint_folder)
    os.mkdir(checkpoint_folder)

    # get fold indices
    train_idx = kfold['train_idx']
    val_idx = kfold['val_idx']
    test_idx = kfold['test_idx']

    # get input features
    X_train = X.loc[train_idx, :].to_numpy()
    X_val = X.loc[val_idx, :].to_numpy()
    X_test = X.loc[test_idx, :].to_numpy()

    # get labels
    Y_train = Y.loc[train_idx, :].to_numpy()
    Y_val = Y.loc[val_idx, :].to_numpy()
    Y_test = Y.loc[test_idx, :]

    # save target labels for analysis
    Y_test.to_csv(f'input_test_{k}.csv', index=False)
    Y_test = Y_test.to_numpy()

    # Normalize input. Nothing happens for binary fingerpritns. Counts gets normalized.
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_val = (X_val - X_train.min()) / (X_train.max() - X_train.min())
    X_test = (X_test - X_train.min()) / (X_train.max() - X_train.min())

    # standardise output
    Y_scaler = StandardScaler()
    Y_train_scaled = Y_scaler.fit_transform(Y_train)
    Y_test_scaled = Y_scaler.transform(Y_test)
    Y_val_scaled = Y_scaler.transform(Y_val)

    # create tensors
    X_train_tensor = torch.tensor(X_train).float()
    X_val_tensor = torch.tensor(X_val).float()
    X_test_tensor = torch.tensor(X_test).float()

    Y_train_tensor = torch.tensor(Y_train_scaled).float()
    Y_val_tensor = torch.tensor(Y_val_scaled).float()
    Y_test_tensor = torch.tensor(Y_test_scaled).float()

    #print(f'  X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}')
    
    input_size = X_train_tensor.size()[-1]
    out_size = Y_train_tensor.size()[-1]

    # create dataset objects
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    # Do hyperparam optimization on validation set if requested
    # This will replace some of the input arguments with the optimized ones
    if hopt > 0:
        
        print("    !!! Running Hyperopt !!!")
        print("    ------------------------")
        print("    Input arguments 'num_hidden_layers', 'learning_rate', 'dropout_rate' will be ignored")
        print("")
        sleep(2)

        # define hyperopt search space
        hspace = {'num_hidden_layers': hp.choice('num_hidden_layers', [1, 2]),
                  'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
                  'dropout_rate': hp.choice('dropout_rate', [0., 0.1, 0.2, 0.3, 0.4, 0.5])
                 }

        fmin_objective = partial(hopt_objective, train_dataset=train_dataset, val_dataset=val_dataset, 
                                 input_size=input_size, out_size=out_size, 
                                 batch_size=batch_size, hidden_size=hidden_size, 
                                 num_epochs=num_epochs, patience=patience, 
                                 checkpoint_folder=checkpoint_folder, device=device)

        best = fmin(fn=fmin_objective, space=hspace, algo=tpe.suggest, trials=Trials(),
                    max_evals=hopt, rstate=np.random.RandomState(42))
    
        # get best params
        best_params = space_eval(hspace, best)
        dropout_rate = best_params['dropout_rate']
        learning_rate = best_params['learning_rate']
        num_hidden_layers = best_params['num_hidden_layers']

    # now train and test
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=0)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # init model
    model = FeedForwardMultiOutput(input_size=input_size, 
                                   out_size=out_size, 
                                   hidden_size=hidden_size, 
                                   num_hidden_layers=num_hidden_layers, 
                                   dropout_rate=dropout_rate)
    print('')
    print('         Model Info')
    print('-'*30)
    print(model)
    model.to(device) # send model to device

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    model, _, _, _, _ = train(device=device, model=model, 
                              num_epochs=num_epochs, optimizer=optimizer, 
                              loss_function=loss_function, 
                              train_loader=train_loader, 
                              valid_loader=val_loader, 
                              patience=patience, 
                              print_interval=1, 
                              checkpoint_folder=checkpoint_folder)

    # predict test set
    model.eval()  # eval mode
    with torch.no_grad():  # deactivate autograd
        Y_pred_tensor = model(X_test_tensor.to(device))
    
    Y_pred_scaled = Y_pred_tensor.cpu().detach().numpy()
    Y_pred = Y_scaler.inverse_transform(Y_pred_scaled)

    df_predictions = pd.DataFrame(np.array(Y_pred), columns=Y.columns)
    df_predictions.to_csv(f'predictions_{k}.csv', index=False)

    # cleanup checkpoints
    if os.path.isdir(checkpoint_folder):
        shutil.rmtree(checkpoint_folder)
