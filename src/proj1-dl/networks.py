import torch
from torch import nn
from torch.nn import functional as F

class FC_net(nn.Module):
    def __init__(self, layers):
        super(FC_net, self).__init__()
        self.additional_hidden = nn.ModuleList()
        for l in range(len(layers) - 1):
            self.additional_hidden.append(nn.Linear(layers[l], layers[l + 1]))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for l in range(len(self.additional_hidden) - 1):
            x = F.relu(self.additional_hidden[l](x))
        x = self.additional_hidden[-1](x)
        return x


class Conv_net(nn.Module):
    def __init__(self, size, layers, layers_conv, kernel_size, pooling_kernel_size, p):
        super(Conv_net, self).__init__()
        self.pooling_kernel_size = pooling_kernel_size
        self.additional_conv_hidden = nn.ModuleList()
        self.additional_fc_hidden = nn.ModuleList()
        self.droput_layers = nn.ModuleList()
        self.batch_normalization = nn.ModuleList()
        self.size = size

        for l in range(len(layers_conv) - 1):
            self.additional_conv_hidden.append(
                nn.Conv1d(layers_conv[l], layers_conv[l + 1], kernel_size=kernel_size[l]))
            self.droput_layers.append(torch.nn.Dropout(p=p[l]))
            self.batch_normalization.append(torch.nn.BatchNorm1d(layers_conv[l + 1]))

        for i in range(len(kernel_size)):
            size-=(kernel_size[i]-1)

            size//=pooling_kernel_size[i]

        self.additional_fc_hidden.append(nn.Linear(size * layers_conv[-1], layers[0]))
        self.droput_layers.append(torch.nn.Dropout(p=p[l + 1]))
        self.batch_normalization.append(torch.nn.BatchNorm1d(layers[0]))
        self.flat_size = size * layers_conv[-1]

        start_p = l + 2

        for l in range(len(layers) - 1):
            self.additional_fc_hidden.append(nn.Linear(layers[l], layers[l + 1]))
            if l != len(layers) - 2:
                self.droput_layers.append(torch.nn.Dropout(p=p[l + start_p]))
                self.batch_normalization.append(torch.nn.BatchNorm1d(layers[l + 1]))

    def forward(self, x):
        for l in range(len(self.additional_conv_hidden)):
            x = self.droput_layers[l](self.batch_normalization[l](
                F.relu(F.max_pool1d(self.additional_conv_hidden[l](x), kernel_size=self.pooling_kernel_size[l]))))
        x = x.view(-1, self.flat_size)
        for l in range(len(self.additional_fc_hidden) - 1):
            index = len(self.additional_conv_hidden) + l
            x = self.droput_layers[index](self.batch_normalization[index](F.relu(self.additional_fc_hidden[l](x))))
        x = self.additional_fc_hidden[-1](x)
        return x