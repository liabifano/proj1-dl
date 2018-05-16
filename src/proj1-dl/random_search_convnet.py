import numpy as np
import pprint
from logging import error
from sklearn import model_selection

import dlc_bci as bci
from networks import Conv_net
from train import train_model_full

N_RANDOM_MODELS = 500
MINI_BATCH_SIZE = 40
N_FOLDS = 10
N_EPOCHS = 500

if __name__ == '__main__':
    """
    Random search for convolutational neural network
    
    1. Download train
    2. Create folds based on train
    3. Run different models, grab random parameters possibilities and print its performance in the screen
    4. Select best model over the interactions 
    """
    train_input, train_target = bci.load(root='./data_bci', one_khz=True)
    kfolds = model_selection.KFold(n_splits=N_FOLDS, random_state=1234, shuffle=True)

    mu, std = train_input.mean(0), train_input.std(0)
    train_input = train_input.sub_(mu).div_(std)

    X = train_input.numpy()
    y = train_target.numpy()

    run_list = []
    errors = 0
    performances = {}
    for i in range(N_RANDOM_MODELS):
        try:
            # param_num_layers_fc
            num_layers_fc = np.random.randint(1, 4)
            # param_num_layers_conv
            num_layers_conv = np.random.randint(1, 4)
            layers_fc = np.random.randint(1, 50, num_layers_fc).tolist() + [2]
            layers_conv = [28] + np.random.randint(1, 20, num_layers_conv).tolist()
            kernel_size = np.random.randint(1, 8, num_layers_conv).tolist()
            pooling_kernel_size = np.random.randint(2, 4, num_layers_conv).tolist()
            p = np.random.rand(num_layers_fc + num_layers_conv).tolist()
            parameters = {'lambda': np.linspace(0.01, 0.1, 30).tolist() + [0],
                          'lr': np.linspace(0.001, 0.1, 50).tolist()}
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
                                                              lambdda=lambdda,
                                                              lr=lr)

            performances[i] = {'params': param,
                               'acc_val': acc_val,
                               'model': model_class.__name__}

        except:
            errors += 1
            error('Fit fail')

    best_position = np.argmax([p['acc_val'][-1] for p in performances.values()])
    best_model = performances[[*performances][best_position]]
    best_model['params']['lambdda'] = best_model['lambdda']
    best_model['params']['lr'] = best_model['lr']
    print('\n\n>>>> End of {} models in random search <<<<'.format(N_RANDOM_MODELS))
    print('{} sucessfull models run'.format(N_RANDOM_MODELS - errors))
    print('{} failed models run\n'.format(errors))
    print('!!!!!!!!!!!!!!!!!!!!!!!! AAAAAAAAANNNNNNNNDDD THE BEST MODEL ISSSSSSS!!!!!')
    print('Best Model is `{}` with accuracy in validation of {} and parameters:'.format(best_model['model'],
                                                                                        round(best_model['acc_val'][-1],
                                                                                              3)))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(best_model['params'])
