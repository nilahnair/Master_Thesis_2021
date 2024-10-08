'''
Created on Mar 28, 2019

@author: fmoya
'''


from __future__ import print_function
import logging


import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import numpy as np


class Network(nn.Module):
    '''
    classdocs
    '''


    def __init__(self, config):
        '''
        Constructor
        '''
        
        super(Network, self).__init__()
        
        logging.info('            Network: Constructor')
        
        self.config = config
        

        # set the Conv layers
        self.conv1_1 = nn.Conv2d(in_channels = 1, out_channels = self.config['num_filters'], kernel_size = (self.config['filter_size'], 1), stride = 1, padding = 0)
        Wx, Hx = self.size_feature_map(Wx = self.config['sliding_window_length'], 
                                       Hx = self.config['NB_sensor_channels'], 
                                       F = (self.config['filter_size'], 1), 
                                       P = 0, S = (1, 1), type_layer = 'conv')
        self.conv1_2 = nn.Conv2d(in_channels = self.config['num_filters'], out_channels = self.config['num_filters'], kernel_size = (self.config['filter_size'], 1), stride = 1, padding = 0)
        Wx, Hx = self.size_feature_map(Wx = Wx, 
                                       Hx = Hx, 
                                       F = (self.config['filter_size'], 1), 
                                       P = 0, S = (1, 1), type_layer = 'conv')
        self.conv2_1 = nn.Conv2d(in_channels = self.config['num_filters'], out_channels = self.config['num_filters'], kernel_size = (self.config['filter_size'], 1), stride = 1, padding = 0)
        Wx, Hx = self.size_feature_map(Wx = Wx, 
                                       Hx = Hx, 
                                       F = (self.config['filter_size'], 1), 
                                       P = 0, S = (1, 1), type_layer = 'conv')
        self.conv2_2 = nn.Conv2d(in_channels = self.config['num_filters'], out_channels = self.config['num_filters'], kernel_size = (self.config['filter_size'], 1), stride = 1, padding = 0)
        Wx, Hx = self.size_feature_map(Wx = Wx, 
                                       Hx = Hx, 
                                       F = (self.config['filter_size'], 1), 
                                       P = 0, S = (1, 1), type_layer = 'conv')
        
        self.fc3 = nn.Linear(self.config['num_filters'] * int(Wx) * self.config['NB_sensor_channels'], 256)
        self.fc4 = nn.Linear(256, 256)
        
        if self.config['output'] == 'softmax': 
            self.fc5 = nn.Linear(256, self.config['num_classes'])
        elif self.config['output'] == 'attribute':  
            self.fc5 = nn.Linear(256, self.config['num_attributes'])
        
        self.softmax = nn.Softmax()
        
        self.sigmoid = nn.Sigmoid()
        
        
        return
    
    
    
    
    
    def forward(self, x):

        if self.config["network"] == "cnn":
            x = F.relu(self.conv1_1(x))
            x = F.relu(self.conv1_2(x))
            #x12 = F.max_pool2d(x12, (2, 1))

            x = F.relu(self.conv2_1(x))
            x = F.relu(self.conv2_2(x))
            # x = F.max_pool2d(x, (2, 1))

            # view is reshape
            x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
            x = F.relu(self.fc3(x))
        elif self.config["network"] == "cnn_imu":
            # LA
            if self.config["reshape_input"]:
                if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                    idx_LA = np.arange(0, 36)
                    idx_LA = np.concatenate([idx_LA, np.arange(63, 72)])
                    idx_LA = np.concatenate([idx_LA, np.arange(72, 81)])
                    x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LA]))
                elif self.config["dataset"] == 'pamap2':
                    idx_LA = np.arange(1, 14)
                    x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LA]))
            else:
                if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                    idx_LA = np.arange(0, 36)
                    idx_LA = np.concatenate([idx_LA, np.arange(63, 72)])
                    idx_LA = np.concatenate([idx_LA, np.arange(72, 81)])
                elif self.config["dataset"] == 'pamap2':
                    idx_LA = np.arange(1, 14)
                    x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LA]))

            x_LA = F.relu(self.conv_LA_1_2(x_LA))
            x_LA = F.relu(self.conv_LA_2_1(x_LA))
            x_LA = F.relu(self.conv_LA_2_2(x_LA))
            # view is reshape
            x_LA = x_LA.view(-1, x_LA.size()[1] * x_LA.size()[2] * x_LA.size()[3])
            x_LA = F.relu(self.fc3_LA(x_LA))

            # LL
            if self.config["reshape_input"]:
                if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                    idx_LL = np.arange(0, 36)
                    idx_LL = np.concatenate([idx_LL, np.arange(85, 97)])
                elif self.config["dataset"] == 'pamap2':
                    idx_LL = np.arange(27, 40)
            else:
                if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                    idx_LL = np.arange(0, 36)
                    idx_LL = np.concatenate([idx_LL, np.arange(85, 97)])
                elif self.config["dataset"] == 'pamap2':
                    idx_LL = np.arange(27, 40)
                    x_LL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LL]))

            x_LL = F.relu(self.conv_LL_1_2(x_LL))
            x_LL = F.relu(self.conv_LL_2_1(x_LL))
            x_LL = F.relu(self.conv_LL_2_2(x_LL))
            # view is reshape
            x_LL = x_LL.view(-1, x_LL.size()[1] * x_LL.size()[2] * x_LL.size()[3])
            x_LL = F.relu(self.fc3_LL(x_LL))

            # N
            if self.config["reshape_input"]:
                if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                    idx_N = np.arange(0, 36)
                    idx_N = np.concatenate([idx_N, np.arange(36, 45)])
                elif self.config["dataset"] == 'pamap2':
                    idx_N = np.arange(0, 1)
                    idx_N = np.concatenate([idx_N, np.arange(14, 27)])
                    x_N = F.relu(self.conv_LA_1_1(x[:, :, :, idx_N]))
            else:
                if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                    idx_N = np.arange(0, 36)
                    idx_N = np.concatenate([idx_N, np.arange(36, 45)])
                elif self.config["dataset"] == 'pamap2':
                    idx_N = np.arange(0, 1)
                    idx_N = np.concatenate([idx_N, np.arange(14, 27)])
                    x_N = F.relu(self.conv_LA_1_1(x[:, :, :, idx_N]))
            x_N = F.relu(self.conv_N_1_2(x_N))
            x_N = F.relu(self.conv_N_2_1(x_N))
            x_N = F.relu(self.conv_N_2_2(x_N))
            # view is reshape
            x_N = x_N.view(-1, x_N.size()[1] * x_N.size()[2] * x_N.size()[3])
            x_N = F.relu(self.fc3_N(x_N))

            # RA
            if self.config["reshape_input"]:
                if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                    idx_RA = np.arange(0, 36)
                    idx_RA = np.concatenate([idx_RA, np.arange(54, 63)])
                    idx_RA = np.concatenate([idx_RA, np.arange(63, 72)])
                elif self.config["dataset"] == 'pamap2':
                    idx_RA = np.arange(1, 14)
                    x_RA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RA]))
            else:
                if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                    idx_RA = np.arange(0, 36)
                    idx_RA = np.concatenate([idx_RA, np.arange(54, 63)])
                    idx_RA = np.concatenate([idx_RA, np.arange(63, 72)])
                elif self.config["dataset"] == 'pamap2':
                    idx_RA = np.arange(1, 14)
                    x_RA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RA]))

            x_RA = F.relu(self.conv_RA_1_2(x_RA))
            x_RA = F.relu(self.conv_RA_2_1(x_RA))
            x_RA = F.relu(self.conv_RA_2_2(x_RA))
            # view is reshape
            x_RA = x_RA.view(-1, x_RA.size()[1] * x_RA.size()[2] * x_RA.size()[3])
            x_RA = F.relu(self.fc3_RA(x_RA))

            # RL
            if self.config["reshape_input"]:
                if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                    idx_RL = np.arange(0, 36)
                    idx_RL = np.concatenate([idx_RL, np.arange(85, 97)])
                elif self.config["dataset"] == 'pamap2':
                    idx_RL = np.arange(27, 40)
                    x_RL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RL]))
            else:
                if self.config["dataset"] == 30:
                    idx_RL = np.arange(0, 36)
                    idx_RL = np.concatenate([idx_RL, np.arange(85, 97)])
                elif self.config["dataset"] == 'pamap2':
                    idx_RL = np.arange(27, 40)
                    x_RL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RL]))

            x_RL = F.relu(self.conv_RL_1_2(x_RL))
            x_RL = F.relu(self.conv_RL_2_1(x_RL))
            x_RL = F.relu(self.conv_RL_2_2(x_RL))
            # view is reshape
            x_RL = x_RL.view(-1, x_RL.size()[1] * x_RL.size()[2] * x_RL.size()[3])
            x_RL = F.relu(self.fc3_RL(x_RL))

            x = torch.cat((x_LA, x_LL, x_N, x_RA, x_RL), 1)
        
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, training=self.training)
        x = self.fc5(x)
        
        if self.config['output'] == 'attribute':
            x = self.sigmoid(x)

        if not self.training:
            if self.config['output'] == 'softmax':
                x = self.softmax(x)

        return x
    
    
    
    def init_weights(self):
        self.apply(Network._init_weights_orthonormal)
        
        return
    
    
    
    @staticmethod
    def _init_weights_orthonormal(m):
        if isinstance(m, nn.Conv2d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
            nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
            nn.init.constant_(m.bias.data, 0)
        if isinstance(m, nn.Linear):          
            nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
            nn.init.constant_(m.bias.data, 0)
        
        return
    
    
    
    
    def size_feature_map(self, Wx, Hx, F, P, S, type_layer = 'conv'):
        
        if type_layer == 'conv':
            Wy = 1 + (Wx - F[0] + 2 * P) / S[0]
            Hy = 1 + (Hx - F[1] + 2 * P) / S[1]
        
        elif type_layer == 'pool':
            Wy = 1 + (Wx - F[0]) / S[0]
            Hy = 1 + (Hx - F[1]) / S[1]
                    
        return Wy, Hy
    
    
    
    
    
    
