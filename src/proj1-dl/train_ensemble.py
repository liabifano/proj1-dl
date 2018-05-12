import numpy as np
import os
import torch
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from torch.autograd import Variable

import dlc_bci as bci
from utils import predict_classes

param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 2),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

TRAIN_MODEL_PATH = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'trained_models')

if __name__ == '__main__':
    models = ['convnet_1', 'convnet_2']
    path_models = [os.path.join(TRAIN_MODEL_PATH, x + '.pth') for x in models]

    train_input, train_target = bci.load(root='./data_bci')
    mu, std = train_input.mean(0), train_input.std(0)
    train_input = Variable(train_input.sub_(mu).div_(std))
    train_target = Variable(train_target)

    test_input, test_target = bci.load(root='./data_bci', train=False)
    test_input = test_input.sub_(mu).div_(std)
    test_input = Variable(test_input)
    test_target = Variable(test_target)

    models = [torch.load(x) for x in path_models]

    preds_train_1 = [predict_classes(m, train_input, 50) for m in models]
    preds_test_1 = [predict_classes(m, test_input, 50) for m in models]

    preds_train_1 = np.transpose(np.vstack(preds_train_1))
    preds_test_1 = np.transpose(np.vstack(preds_test_1))

    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, preds_train_1.shape[1]),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    clf = RandomForestClassifier(n_estimators=20)
    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_dist,
                                       n_iter=1000)
    random_search.fit(preds_train_1, train_target.data.numpy())
    best_clf = random_search.best_estimator_
    final_preds = best_clf.predict(preds_test_1)

    print(accuracy_score(test_target.data.numpy(), final_preds))
