import numpy as np
import pprint
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

from utils import compute_nb_errors


def train_model_full(network_model,
                     param,
                     X, y,
                     mini_batch_size,
                     kfolds,
                     nb_epochs,
                     lambdda=0.01,
                     lr=0.001):
    acc_train_kfold = []
    loss_train_kfold = []
    acc_val_kfold = []
    loss_val_kfold = []

    for train_index, val_index in kfolds.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train = Variable(torch.from_numpy(X_train))
        X_val = Variable(torch.from_numpy(X_val))
        y_train = Variable(torch.from_numpy(y_train))
        y_val = Variable(torch.from_numpy(y_val))

        model = network_model(**param)

        loss_train, loss_val, acc_train, acc_val = train_model(model,
                                                               X_train, y_train,
                                                               X_val, y_val,
                                                               mini_batch_size,
                                                               nb_epochs,
                                                               lambdda, lr)
        acc_train_kfold.append(acc_train)
        loss_train_kfold.append(loss_train)
        acc_val_kfold.append(acc_val)
        loss_val_kfold.append(loss_val)

    acc_train_kfold = np.mean(np.array(acc_train_kfold), axis=0)
    acc_val_kfold = np.mean(np.array(acc_val_kfold), axis=0)

    loss_train_kfold = np.mean(np.array(loss_train_kfold), axis=0)
    loss_val_kfold = np.mean(np.array(loss_val_kfold), axis=0)

    pp = pprint.PrettyPrinter(indent=4)
    print('{} epochs done for `{}` with the parameters:'.format(nb_epochs, network_model.__name__))
    pp.pprint(param)
    print('\n   Loss: train ~ {} Acc train ~ {} \n   Loss: val ~ {} / Acc val ~ {}'
          .format(round(loss_train_kfold[-1], 3),
                  round(acc_train_kfold[-1], 3),
                  round(loss_val_kfold[-1], 3),
                  round(acc_val_kfold[-1], 3)))
    print('----------------------------------------------------------------')
    return loss_train_kfold, loss_val_kfold, acc_train_kfold, acc_val_kfold


def train_model(model, X_train, y_train, X_val, y_val, mini_batch_size, nb_epochs, lambdda=0.01, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    acc_train = []
    acc_val = []
    loss_train = []
    loss_val = []

    for e in range(0, nb_epochs):

        model.train(True)
        for b in list(range(0, X_train.size(0), mini_batch_size)):
            if b + mini_batch_size <= X_train.size(0):
                output = model(X_train.narrow(0, b, mini_batch_size))
                loss = criterion(output, y_train.narrow(0, b, mini_batch_size))
            else:
                output = model(X_train.narrow(0, b, X_train.size(0) - b))
                loss = criterion(output, y_train.narrow(0, b, X_train.size(0) - b))

            for p in model.parameters():
                loss += lambdda * p.pow(2).sum()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            model.train(False)
            output_train = model(X_train)
            output_val = model(X_val)

        acc_val.append(1 - compute_nb_errors(model, X_val, y_val, mini_batch_size=mini_batch_size) / X_val.size(0))
        acc_train.append(
            1 - compute_nb_errors(model, X_train, y_train, mini_batch_size=mini_batch_size) / X_train.size(0))
        loss_train.append(criterion(output_train, y_train).data[0])
        loss_val.append(criterion(output_val, y_val).data[0])

    return loss_train, loss_val, acc_train, acc_val


def train_parallel(param):
    train_model_full(**param)
