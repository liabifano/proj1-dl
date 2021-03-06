import os
import sys
import torch
from sklearn import model_selection
from torch.autograd import Variable

import dlc_bci as bci
from networks import Conv_net
from train import train_model

MINI_BATCH_SIZE = 40
N_FOLDS = 10
N_EPOCHS = 1000
TRAIN_MODEL_PATH = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'trained_models')

if __name__ == '__main__':
    """
    Train Convolutational Network 
    
    1. Download train
    2. Select validation set
    3. Fit neural network in the train set
    4. Evaluate the model in the validation set
    5. Save the model if necessary
    """
    train_input, train_target = bci.load(root='./data_bci')

    mu, std = train_input.mean(0), train_input.std(0)
    train_input = train_input.sub_(mu).div_(std)

    X = train_input.numpy()
    y = train_target.numpy()

    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                                      test_size=0.1,
                                                                      random_state=4321)

    X_train = Variable(torch.from_numpy(X_train))
    X_val = Variable(torch.from_numpy(X_val))
    y_train = Variable(torch.from_numpy(y_train))
    y_val = Variable(torch.from_numpy(y_val))

    params = {   'kernel_size': [1, 2],
                 'layers': [15, 2],
                 'layers_conv': [28, 14, 16],
                 'p': [0.10271629346726219, 0.1647674882927611, 0.768123866247771],
                 'pooling_kernel_size': [3, 3],
                 'size': 50}

    model = Conv_net(**params)

    costs, costs_val, acc, acc_val, model = train_model(model,
                                                        X_train,
                                                        y_train,
                                                        X_val,
                                                        y_val,
                                                        MINI_BATCH_SIZE,
                                                        N_EPOCHS,
                                                        lambdda=0.01, lr=0.001,
                                                        verbose=True,
                                                        early_stop=True)

    import pdb; pdb.set_trace()

    if len(sys.argv) > 2 and sys.argv[1] == 'saveme':
        path_to_save = os.path.join(TRAIN_MODEL_PATH, sys.argv[2] + '.pth')
        torch.save(model, path_to_save)
        print('The model is saved in {}'.format(path_to_save))
