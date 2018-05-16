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
                     lr=0.001,
                     verbose=False):
    """
    Train k neural networks, where k is the number of folds

    :param network_model: class with the neural network
    :param param: parameters to instanciate the network_model
    :param X: array with input
    :param y: array with labels
    :param mini_batch_size: integer with mini batch size
    :param kfolds: list of lists with folds positions for train / validation
    :param nb_epochs: integer with number of epochs
    :param lambdda: lambda parameter to run the neural network
    :param lr: learning rate to run the neural network
    :param verbose: boolean if it is needed to be verbose
    :return: 4 lists with {avg loss in train, avg loss in validation, avg accuracy in train, avg accuracy in validation}
    """
    acc_train_kfold = []
    loss_train_kfold = []
    acc_val_kfold = []
    loss_val_kfold = []

    for d, (train_index, val_index) in enumerate(kfolds.split(X)):
        if verbose:
            print('\nFold {}'.format(d))
            print('----------------------------------------------------------------')

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train = Variable(torch.from_numpy(X_train))
        X_val = Variable(torch.from_numpy(X_val))
        y_train = Variable(torch.from_numpy(y_train))
        y_val = Variable(torch.from_numpy(y_val))

        model = network_model(**param)

        loss_train, loss_val, acc_train, acc_val, _ = train_model(model,
                                                                  X_train, y_train,
                                                                  X_val, y_val,
                                                                  mini_batch_size,
                                                                  nb_epochs,
                                                                  lambdda, lr,
                                                                  verbose)
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


def train_model(model, X_train, y_train,
                X_val, y_val,
                mini_batch_size, nb_epochs,
                lambdda=0.01, lr=0.001,
                early_stop=False,
                verbose=False):
    """
    Fit neural network and evaluate it in a validation set

    :param model: neural network object
    :param X_train: tensor with inputs of train
    :param y_train: tensor with outputs of train
    :param X_val: tensor with inputs of validation
    :param y_val: tensor with outputs of validation
    :param mini_batch_size: integer with mini batch size
    :param nb_epochs: integer with the number of epochs
    :param lambdda: lambda parameter to run the neural network
    :param lr: learning rate to run the neural network
    :param early_stop: if it is needed early stop
    :param verbose: boolean if it is needed to be verbose
    :return: 4 lists with {loss in train, loss in validation, accuracy in train, accuracy in validation} and fitted model
    """
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

        if verbose and (e - 1) % 100 == 0:
            print(
            'Epoch {}, accuracy in validation: {} / train {}'.format(e, round(acc_val[-1], 3), round(acc_train[-1], 3)))

        if early_stop:
            # if len(acc_val) > 2 and acc_val[-1] - acc_val[-2] < -0.001:
            #     print('! Ops! Something when wrong...\nEarly stop in epoch {}'.format(e))
            #     break
            pass

    return loss_train, loss_val, acc_train, acc_val, model


def train_parallel(param):
    train_model_full(**param)
