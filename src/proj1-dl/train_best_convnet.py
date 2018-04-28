from sklearn import model_selection
from torch.autograd import Variable
import torch

import dlc_bci as bci
from networks import Conv_net
from train import train_model

MINI_BATCH_SIZE = 40
N_FOLDS = 10
N_EPOCHS = 1000

if __name__ == '__main__':
    train_input, train_target = bci.load(root='./data_bci')
    kfolds = model_selection.KFold(n_splits=N_FOLDS, random_state=1234, shuffle=True)

    test_input, test_target = bci.load(root='./data_bci', train=False)

    # put this inside the train to avoid data snooping
    mu, std = train_input.mean(0), train_input.std(0)
    train_input = train_input.sub_(mu).div_(std)
    test_input = test_input.sub_(mu).div_(std)

    X = train_input.numpy()
    y = train_target.numpy()

    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                                        test_size=0.1,
                                                                        random_state=4321)

    X_train = Variable(torch.from_numpy(X_train))
    X_val = Variable(torch.from_numpy(X_val))
    y_train = Variable(torch.from_numpy(y_train))
    y_val = Variable(torch.from_numpy(y_val))

    params = {   'kernel_size': [2],
                 'layers': [22, 2],
                 'layers_conv': [28, 19],
                 'p': [0.8235148994463269, 0.5220469611224193],
                 'pooling_kernel_size': [2],
                 'size': 50}

    model = Conv_net(**params)

    costs, costs_val, acc, acc_val = train_model(model,
                                                 X_train,
                                                 y_train,
                                                 X_val,
                                                 y_val,
                                                 MINI_BATCH_SIZE,
                                                 N_EPOCHS,
                                                 lambdda=0.014210526315789474,
                                                 lr=0.007061224489795919,
                                                 verbose=True,
                                                 early_stop=True)
