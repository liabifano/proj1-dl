import os
import sys
import torch
from torch.autograd import Variable

import dlc_bci as bci
from utils import compute_nb_errors

TRAIN_MODEL_PATH = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'trained_models')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('You must specify the saved model name')
        sys.exit(1)

    train_input, train_target = bci.load(root='./data_bci')
    # put this inside the train to avoid data snooping
    mu, std = train_input.mean(0), train_input.std(0)

    test_input, test_target = bci.load(root='./data_bci', train=False)
    test_input = test_input.sub_(mu).div_(std)
    test_input = Variable(test_input)
    test_target = Variable(test_target)


    path_to_consume = os.path.join(TRAIN_MODEL_PATH, sys.argv[1])
    print('Reading from {}'.format(path_to_consume))

    trained_model = torch.load(path_to_consume)
    pred_test = trained_model(test_input)

    error = compute_nb_errors(trained_model, test_input, test_target,
                              mini_batch_size=test_input.size(0)) / test_input.size(0)

    print('Performance of {} in test is: {}'.format(sys.argv[1], 1-error))
