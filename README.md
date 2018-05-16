# Mini Project 1 - EE559

Implement a neural network to predict the laterality of finger movement
(left or right) from the EEG recording

### Bootstrap python env
```bash
bash bootstrap-python-env.sh
```

### Activate python env
```bash
source activate deep
```

### To run and get the final performance
```bash
source activate deep
python src/proj1-dl/train_ensemble_and_score.py 
```


### File by file
`src/proj1-dl/dlc_bci.py:` download data, copied from course webpage

`src/proj1-dl/network.py:` all network designs tried in the project

`src/proj1-dl/train.py:` help functions to performance random search

`src/proj1-dl/random_search_convnet.py:` performs random search for convolutional networks

`src/proj1-dl/random_search_lstm.py:` performs random search for lstm

`src/proj1-dl/train_best_convnet.py:` train and saves convolutional networks

`src/proj1-dl/train_dummy_net.py:` train the simplest model, neural network with one layer

`src/proj1-dl/train_ensemble_and_score.py:` loads models, performs random search in Random Forest and score the test set

`src/proj1-dl/score_test.py:` loads a saved model and score the test set

`src/proj1-dl/utils.py:` help functions to evaluate performance

### Data Analysis and Interaction process
`src/proj1-dl/data-analysis.ipynb:` first data analysis to have a feeling of the data

`src/proj1-dl/cross-validation.ipynb:` build cross validation chart based on convolutional neural network

`src/proj1-dl/LSTM.ipynb:` draft of LSTM model

`src/proj1-dl/LSTM.ipynb:` draft of Convolutional Network model





 