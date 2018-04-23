
# coding: utf-8

# In[1]:


import torch
from torch import Tensor
import numpy as np
from torch import nn
import dlc_bci as bci
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from multiprocessing import Pool
get_ipython().magic('matplotlib inline')

from sklearn import model_selection


# ### 0. Data loading and preprocessing

# #### Data loading

# In[2]:


train_input, train_target = bci.load(root = './data_bci')
print(str(type(train_input)), train_input.size()) 
print(str(type(train_target)), train_target.size())
X = train_input.numpy()
y = train_target.numpy()
kfolds = model_selection.KFold(n_splits=10, random_state=1234, shuffle=True)


# In[3]:


test_input , test_target = bci.load(root = './data_bci', train = False)
print(str(type(test_input)), test_input.size()) 
print(str(type(test_target)), test_target.size())


# #### Data normalization

# In[4]:


# put this inside the train to avoid data snooping
mu, std = train_input.mean(0), train_input.std(0)
train_input.sub_(mu).div_(std)
test_input.sub_(mu).div_(std)
print("Normalization is done!")


# In[5]:


test_input  = Variable(test_input)
test_target = Variable(test_target)


# #### Utility functions

# In[6]:


def train_model_full(param, train_input, train_target, kfolds, nb_epochs, lambdda = 0.01, lr = 0.001):
    
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
        
        if param['type']=='fc': model = FC_net(param['layers'])
        elif param['type']=='conv': model = model_conv = Conv_net(param['layers'], param['layers_conv'],                                                     param['kernel_size'], param['pooling_kernel_size'], param['p_list'])
        
        loss_train, loss_val, acc_train, acc_val = train_model(model, X_train, y_train, X_val, y_val, kfolds, nb_epochs, lambdda, lr)
        acc_train_kfold.append(acc_train)
        loss_train_kfold.append(loss_train)
        acc_val_kfold.append(acc_val)
        loss_val_kfold.append(loss_val)
        
    acc_train_kfold = np.mean(np.array(acc_train_kfold), axis=0)
    acc_val_kfold = np.mean(np.array(acc_val_kfold), axis=0)

    loss_train_kfold = np.mean(np.array(loss_train_kfold), axis=0)
    loss_val_kfold = np.mean(np.array(loss_val_kfold), axis=0)
    
    print('\n\n---- Epochs Done -----\n')
    print('Loss: train ~ {} Acc train ~ {} \n   Loss: val ~ {} / Acc val ~ {}\n'
         .format(round(loss_train_kfold[-1], 3), 
                 round(acc_train_kfold[-1], 3), 
                 round(loss_val_kfold[-1], 3), 
                 round(acc_val_kfold[-1], 3)))
    return loss_train_kfold, loss_val_kfold, acc_train_kfold, acc_val_kfold


# In[7]:


def train_model(model, X_train, y_train, X_val, y_val, kfolds, nb_epochs, lambdda = 0.01, lr = 0.001):
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
                loss += lambdda*p.pow(2).sum()
            model.zero_grad()
            loss.backward()
            optimizer.step()
                
            model.train(False)
            output_train = model(X_train)
            output_val = model(X_val)
            
        acc_val.append(1-compute_nb_errors(model, X_val, y_val, mini_batch_size=mini_batch_size)/X_val.size(0))
        acc_train.append(1-compute_nb_errors(model, X_train, y_train, mini_batch_size=mini_batch_size)/X_train.size(0))
        loss_train.append(criterion(output_train, y_train).data[0])
        loss_val.append(criterion(output_val, y_val).data[0])
        
#         if (e % 100 == 0):
#                 print('Epoch {}: \n   CVLoss: train ~ {} CVAcc train ~ {} \n   CVLoss: val ~ {} / CVAcc val ~ {}'
#                   .format(e, 
#                           round(loss_train_epoch[-1], 3),
#                           round(acc_train_epoch[-1], 3), 
#                           round(loss_val_epoch[-1], 3),
#                           round(acc_val_epoch[-1], 3)))
        
#     print('\n\n---- Epochs Done -----\n')
#     print('CVLoss: train ~ {} CVAcc train ~ {} \n   CVLoss: val ~ {} / CVAcc val ~ {}'
#          .format(round(loss_train_epoch[-1], 3), 
#                  round(acc_train_epoch[-1], 3), 
#                  round(loss_val_epoch[-1], 3), 
#                  round(acc_val_epoch[-1], 3)))

    return loss_train, loss_val, acc_train, acc_val


# In[8]:


def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        if b + mini_batch_size <= data_input.size(0):
            output = model(data_input.narrow(0, b, mini_batch_size))
            _, predicted_classes = torch.max(output.data, 1)
            for k in range(0, mini_batch_size):
                if data_target.data[b + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1
        else:       
            output = model(data_input.narrow(0, b, data_input.size(0) - b))
            _, predicted_classes = torch.max(output.data, 1)
            for k in range(0, data_input.size(0) - b):
                if data_target.data[b + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1

    return nb_data_errors


# ### 1. Linear model

# ### 2. Fully connected model

# #### Create network

# In[9]:


class FC_net(nn.Module):
    def __init__(self, layers):
        super(FC_net, self).__init__() 
        self.additional_hidden = nn.ModuleList()
        for l in range(len(layers)-1):
            self.additional_hidden.append(nn.Linear(layers[l], layers[l+1]))

    def forward(self, x):
        x=x.view(x.shape[0], -1)
        for l in range(len(self.additional_hidden)-1):
            x = F.relu(self.additional_hidden[l](x))
        x = self.additional_hidden[-1](x)
        return x


# #### Train network

# In[10]:


# check model parameters
layers = [train_input.view(train_input.shape[0], -1).shape[1], 5, 5, 2]
# for k in model_fc.parameters():
#     print(k.size())
parameters = {'type': 'fc', 'layers': layers}
    
mini_batch_size = 42
nb_epochs = 10
#costs, costs_val, acc, acc_val = train_model_full(parameters, train_input, train_target, kfolds, nb_epochs, lambdda=0.02)


# In[11]:


#plot learning curves
# fig, (ax1, ax2) = plt.subplots(1,2)
# ax1.plot (range(nb_epochs), costs)
# ax1.plot (range(nb_epochs), costs_val)
# ax2.plot (range(nb_epochs), acc)
# ax2.plot (range(nb_epochs), acc_val)


# #### Assess network

# In[12]:


#print('train_error {:.02f}%'.format(
#            compute_nb_errors(model_fc, Variable(train_input), Variable(train_target), mini_batch_size = 79) / train_input.size(0) * 100))
#print('test_error {:.02f}%'.format(
#            compute_nb_errors(model_fc, test_input, test_target, mini_batch_size = 20) / test_input.size(0) * 100))

# print("train data error = {}/316 %".format(compute_nb_errors(model, train_input, train_target))
# compute_nb_errors(model, test_input, test_target)


# In[13]:


#train model few times and take average, because of different initialization
#figure out the case when is 50% error for both


# ### 3. Convolutional neural network

# In[14]:


class Conv_net(nn.Module):
    def __init__(self, layers, layers_conv, kernel_size, pooling_kernel_size, p):
        super(Conv_net, self).__init__()
        self.pooling_kernel_size = pooling_kernel_size
        self.additional_conv_hidden = nn.ModuleList()
        self.additional_fc_hidden = nn.ModuleList()
        self.droput_layers = nn.ModuleList()
        self.batch_normalization = nn.ModuleList()
        
        for l in range(len(layers_conv)-1):
            self.additional_conv_hidden.append(nn.Conv1d(layers_conv[l], layers_conv[l+1], kernel_size=kernel_size[l]))
            self.droput_layers.append(torch.nn.Dropout(p=p[l]))
            self.batch_normalization.append(torch.nn.BatchNorm1d(layers_conv[l+1]))
        size = train_input.shape[2]

        for i in range(len(kernel_size)):
            size-=(kernel_size[i]-1)

            size//=pooling_kernel_size[i]

        self.additional_fc_hidden.append(nn.Linear(size*layers_conv[-1], layers[0]))
        self.droput_layers.append(torch.nn.Dropout(p=p[l+1]))
        self.batch_normalization.append(torch.nn.BatchNorm1d(layers[0]))
        self.flat_size = size*layers_conv[-1]
        
        start_p = l+2

        for l in range(len(layers)-1):
            self.additional_fc_hidden.append(nn.Linear(layers[l], layers[l+1]))
            if l != len(layers)-2:
                self.droput_layers.append(torch.nn.Dropout(p=p[l+start_p]))
                self.batch_normalization.append(torch.nn.BatchNorm1d(layers[l+1]))

    def forward(self, x):
        for l in range(len(self.additional_conv_hidden)):
            x = self.droput_layers[l](self.batch_normalization[l](F.relu(F.max_pool1d(self.additional_conv_hidden[l](x),                                                           kernel_size=self.pooling_kernel_size[l]))))
        x=x.view(-1, self.flat_size)
        for l in range(len(self.additional_fc_hidden)-1):
            index = len(self.additional_conv_hidden)+l
            x = self.droput_layers[index](self.batch_normalization[index](F.relu(self.additional_fc_hidden[l](x))))
        x = self.additional_fc_hidden[-1](x)
        return x


# #### Train network

# In[15]:


p_list = [0.2, 0.2, 0]
layers = [5, 2]
layers_conv = [28, 4, 4]
kernel_size = [6, 6]
pooling_kernel_size = [3, 2]
# for k in model_conv.parameters():
#     print(k.size())
    
parameters = {'type': 'conv', 'layers': layers, 'layers_conv':layers_conv, 'kernel_size': kernel_size,               'pooling_kernel_size': pooling_kernel_size, 'p_list': p_list}
    
mini_batch_size = 79
nb_epochs = 500
#costs, costs_val, acc, acc_val = train_model_full(parameters, train_input, train_target, kfolds, nb_epochs, lambdda=0.0375)
                                             


# In[16]:


#plot learning curves
# fig, (ax1, ax2) = plt.subplots(1,2)
# ax1.plot (range(nb_epochs), costs)
# ax1.plot (range(nb_epochs), costs_val)
# ax2.plot (range(nb_epochs), acc)
# ax2.plot (range(nb_epochs), acc_val)


# In[17]:


# print('train_error {:.02f}%'.format(
#             compute_nb_errors(model_conv, Variable(train_input), Variable(train_target), mini_batch_size = 79) / train_input.size(0) * 100))
# print('test_error {:.02f}%'.format(
#             compute_nb_errors(model_conv, test_input, test_target, mini_batch_size = 20) / test_input.size(0) * 100))


# ### Random hyperparameter search

# In[18]:


parameters = { 
     'lambda': np.linspace(0.01, 0.1, 30).tolist()+[0], 
     'lr': np.linspace(0.001, 0.1, 50).tolist()
}
nb_epochs = 2


# In[22]:


def train_parallel(param):
    train_model_full(**param)


# In[23]:


run_list = []
for i in range (100):
    try:
        # param_num_layers_fc
        num_layers_fc = np.random.randint(1, 4)
        #param_num_layers_conv
        num_layers_conv = np.random.randint(1, 10)
        layers_fc = np.random.randint(1, 50, num_layers_fc).tolist()+[2]
        layers_conv = [28] + np.random.randint(1,20, num_layers_conv).tolist()
        kernel_size = np.random.randint(1, 8, num_layers_conv).tolist()
        pooling_kernel_size = np.random.randint(2, 4, num_layers_conv).tolist()
        p = np.random.rand(num_layers_fc+num_layers_conv).tolist()
        lambdda = parameters['lambda'][np.random.randint(1, len(parameters['lambda']))]
        lr = parameters['lr'][np.random.randint(1, len(parameters['lr']))]
        print("num_layers_fc: {}\nnum_layers_conv: {}\nlayers_fc: {}\nlayers_conv: {}\nkernel_size: {}\n    pooling_kernel_size: {}\np: {}\nlambdda: {}\nlr: {}"              .format(num_layers_fc, num_layers_conv, layers_fc, layers_conv, kernel_size,                       pooling_kernel_size,p, lambdda, lr))
        param = {'type': 'conv', 'layers': layers_fc, 'layers_conv':layers_conv, 'kernel_size': kernel_size,                   'pooling_kernel_size': pooling_kernel_size, 'p_list': p}
        run_params = {'param': param, 'train_input': train_input, 'train_target': train_target, 'kfolds': kfolds, 'nb_epochs': nb_epochs, 'lambdda': lambdda, 'lr': lr}
        run_list.append(run_params)
        #loss_train_kfold, loss_val_kfold, acc_train_kfold, acc_val_kfold = train_model_full(param, train_input, train_target, kfolds, nb_epochs, lambdda, lr)
    except: pass 


# In[24]:


# In[ ]:


#add feature with time
#steap and flat for right/left
#throw away some features, i.e. feature 20


# In[3]:

