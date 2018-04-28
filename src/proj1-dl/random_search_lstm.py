import numpy as np
import pprint
from logging import error
from sklearn import model_selection

import dlc_bci as bci
from networks import RNN
from train import train_model_full

N_RANDOM_MODELS = 2
MINI_BATCH_SIZE = 40
N_FOLDS = 10
N_EPOCHS = 100

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

    run_list = []
    errors = 0
    performances = {}
    for i in range(N_RANDOM_MODELS):
        try:
            p = np.random.rand(1).tolist()[0]
            hidden_layer = int(np.random.choice([28, 128, 256]))
            parameters = {'lambda': np.linspace(0.01, 0.1, 30).tolist() + [0],
                          'lr': np.linspace(0.001, 0.1, 50).tolist()}
            lambdda = parameters['lambda'][np.random.randint(1, len(parameters['lambda']))]
            lr = parameters['lr'][np.random.randint(1, len(parameters['lr']))]

            param = {'output_size': 1,
                     'input_size': train_input.shape[2],
                     'dropout': p,
                     'hidden_size': hidden_layer}

            model_class = RNN

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
