import numpy as np
import pprint
import torch
from sklearn import model_selection
from torch import nn
from torch import optim
from torch.autograd import Variable
from logging import error

import dlc_bci as bci
from networks import Conv_net
from utils import compute_nb_errors


N_RANDOM_MODELS = 100
MINI_BATCH_SIZE = 20
N_FOLDS = 10
N_EPOCHS = 10


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


if __name__ == '__main__':
    train_input, train_target = bci.load(root='./data_bci')
    X = train_input.numpy()
    y = train_target.numpy()
    kfolds = model_selection.KFold(n_splits=N_FOLDS, random_state=1234, shuffle=True)

    test_input, test_target = bci.load(root='./data_bci', train=False)

    # put this inside the train to avoid data snooping
    mu, std = train_input.mean(0), train_input.std(0)
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    run_list = []
    errors = 0
    performances = {}
    for i in range(N_RANDOM_MODELS):
        try:
            # param_num_layers_fc
            num_layers_fc = np.random.randint(1, 4)
            # param_num_layers_conv
            num_layers_conv = np.random.randint(1, 10)
            layers_fc = np.random.randint(1, 50, num_layers_fc).tolist() + [2]
            layers_conv = [28] + np.random.randint(1, 20, num_layers_conv).tolist()
            kernel_size = np.random.randint(1, 8, num_layers_conv).tolist()
            pooling_kernel_size = np.random.randint(2, 4, num_layers_conv).tolist()
            p = np.random.rand(num_layers_fc + num_layers_conv).tolist()
            parameters = {
                'lambda': np.linspace(0.01, 0.1, 30).tolist()+[0],
                'lr': np.linspace(0.001, 0.1, 50).tolist()
            }
            lambdda = parameters['lambda'][np.random.randint(1, len(parameters['lambda']))]
            lr = parameters['lr'][np.random.randint(1, len(parameters['lr']))]

            param = {'layers': layers_fc,
                     'layers_conv': layers_conv,
                     'kernel_size': kernel_size,
                     'pooling_kernel_size': pooling_kernel_size,
                     'p': p,
                     'size': train_input.shape[2]}

            model_class = Conv_net

            costs, costs_val, acc, acc_val = train_model_full(model_class,
                                                              param,
                                                              X,
                                                              y,
                                                              MINI_BATCH_SIZE,
                                                              kfolds,
                                                              N_EPOCHS,
                                                              lambdda=0.0375)

            performances[i] = {'params': param,
                               'acc_val': acc_val,
                               'model': model_class.__name__}

        except:
            errors += 1
            error('Fit fail')

    best_position = np.argmax([p['acc_val'][-1] for p in performances.values()])
    best_model = performances[[*performances][best_position]]
    print('\n\n>>>> End of {} models in random search <<<<'.format(N_RANDOM_MODELS))
    print('{} sucessfull models run'.format(N_RANDOM_MODELS-errors))
    print('{} failed models run\n'.format(errors))
    print('!!!!!!!!!!!!!!!!!!!!!!!! AAAAAAAAANNNNNNNNDDD THE BEST MODEL ISSSSSSS!!!!!')
    print('Best Model is `{}` with accuracy in validation of {} and parameters:'.format(best_model['model'], round(best_model['acc_val'][-1], 3)))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(best_model['params'])



    # test_input = Variable(test_input)
    # test_target = Variable(test_target)

    # parameters = {
    #     'lambda': np.linspace(0.01, 0.1, 30).tolist() + [0],
    #     'lr': np.linspace(0.001, 0.1, 50).tolist()
    # }
    #
    # p_list = [0.2, 0.2, 0]
    # layers = [5, 2]
    # layers_conv = [28, 4, 4]
    # kernel_size = [6, 6]
    # pooling_kernel_size = [3, 2]
    # p_list = [0.2, 0.2, 0]
    #
    # parameters = {'size': train_input.shape[2],
    #               'layers': layers,
    #               'layers_conv': layers_conv,
    #               'kernel_size': kernel_size,
    #               'pooling_kernel_size': pooling_kernel_size,
    #               'p': p_list}
    #
    # costs, costs_val, acc, acc_val = train_model_full(Conv_net,
    #                                                   parameters,
    #                                                   X,
    #                                                   y,
    #                                                   MINI_BATCH_SIZE,
    #                                                   kfolds,
    #                                                   N_EPOCHS,
    #                                                   lambdda=0.0375)