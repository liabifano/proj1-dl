from sklearn import model_selection

import dlc_bci as bci
from networks import Conv_net
from train import train_model_full

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

    costs, costs_val, acc, acc_val = train_model_full(Conv_net,
                                                      {   'kernel_size': [4],
                                                          'layers': [40, 2],
                                                          'layers_conv': [28, 15],
                                                          'p': [0.10799015707321313, 0.24230368007875924],
                                                          'pooling_kernel_size': [2],
                                                          'size': 50},
                                                      X,
                                                      y,
                                                      MINI_BATCH_SIZE,
                                                      kfolds,
                                                      N_EPOCHS,
                                                      lambdda=0.0375)
