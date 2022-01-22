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
from tpp import TPP
import math

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
        self.tpp = TPP(config)
        

        if self.config["reshape_input"]:
            in_channels = 3
            Hx = int(self.config['NB_sensor_channels'] / 3)
        else:
            in_channels = 1
            Hx = self.config['NB_sensor_channels']
        Wx = self.config['sliding_window_length']

        if self.config["dataset"] in ['locomotion', 'gesture'] and self.config["pooling"] in [1, 2, 3, 4]:
            padd = [2, 0]
        else:
            if self.config["aggregate"] == "FCN":
                padd = [2, 0]
            elif self.config["aggregate"] == "FC":
                padd = 0
            elif self.config["aggregate"] == "LSTM":
                padd = [2, 0]
        # Computing the size of the feature maps
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padd, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padd, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        if self.config["pooling"] in [1, 2, 3, 4]:
            Wx = int(Wx / 2) - 1
            Wxp1 = Wx
            self.pooling_Wx = [Wxp1]
            logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padd, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padd, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        if self.config["pooling"] in [2, 4]:
            Wx = int(Wx / 2) - 1
            Wxp2 = Wx
            self.pooling_Wx = [Wxp1, Wxp2]
            logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))

        # set the Conv layers
        if self.config["network"] in ["cnn", "cnn_tpp"]:
            self.conv1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padd)
            self.conv1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padd)
            self.conv2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padd)
            self.conv2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padd)

            if self.config["aggregate"] == "FCN":
                if self.config["reshape_input"]:
                    self.fc3 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
                else:
                    self.fc3 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=256, kernel_size=(1, 1), stride=1, padding=0)

            elif self.config["aggregate"] == "FC":
                if self.config["network"] == "cnn_tpp":
                    self.fc3 = nn.Linear(self.tpp.get_output(), 256)
                else:
                    if self.config["reshape_input"]:
                        self.fc3 = nn.Linear(self.config['num_filters'] *
                                             int(Wx) * int(self.config['NB_sensor_channels'] / 3), 256)
                    else:
                        self.fc3 = nn.Linear(self.config['num_filters'] * int(Wx) * self.config['NB_sensor_channels'],
                                             256)
            elif self.config["aggregate"] == "LSTM":
                if self.config["reshape_input"]:
                    self.fc3 = nn.LSTM(input_size=self.config['num_filters'] *
                                                    int(self.config['NB_sensor_channels'] / 3), hidden_size=256,
                                       batch_first=True, bidirectional=True)
                else:
                    self.fc3 = nn.LSTM(input_size=self.config['num_filters'] * self.config['NB_sensor_channels'],
                                       hidden_size=256, batch_first=True, bidirectional=True)

        # set the Conv layers
        if self.config["network"] in ["cnn_imu", "cnn_imu_tpp"]:
            # LA
            self.conv_LA_1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padd)

            self.conv_LA_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            self.conv_LA_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            self.conv_LA_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            if self.config["aggregate"] == "FCN":
                self.fc3_LA = nn.Conv2d(in_channels=self.config['num_filters'],
                                        out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["aggregate"] == "LSTM":
                if self.config["reshape_input"]:
                    if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                        self.fc3_LA = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 15),
                                           hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["dataset"] == 'pamap2':
                        self.fc3_LA = nn.LSTM(input_size=self.config['num_filters'] * 10,
                                           hidden_size=256, batch_first=True, bidirectional=True)
                else:
                    if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                        self.fc3_LA = nn.LSTM(input_size=self.config['num_filters'] * 54,
                                           hidden_size=256, batch_first=True, bidirectional=True)
                    elif self.config["dataset"] == 'pamap2':
                        self.fc3_LA = nn.LSTM(input_size=self.config['num_filters'] * 13,
                                           hidden_size=256, batch_first=True, bidirectional=True)
            elif self.config["aggregate"] == "FC":
                if self.config["network"] == "cnn_imu_tpp":
                    self.fc3_LA = nn.Linear(self.tpp.get_output(), 256)
                else:
                    if self.config["reshape_input"]:
                        if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                            self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 15), 256)
                        elif self.config["dataset"] == 'pamap2':
                            self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) * 10, 256)
                    else:
                        if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                            self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) * 54, 256)
                        elif self.config["dataset"] == 'pamap2':
                            self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) * 13, 256)



            # LL
            self.conv_LL_1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padd)

            self.conv_LL_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            self.conv_LL_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            self.conv_LL_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            if self.config["aggregate"] == "FCN":
                self.fc3_LL = nn.Conv2d(in_channels=self.config['num_filters'],
                                        out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["aggregate"] == "LSTM":
                if self.config["reshape_input"]:
                    if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                        self.fc3_LL = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 15), hidden_size=256,
                                              batch_first=True, bidirectional=True)
                    elif self.config["dataset"] == 'pamap2':
                        self.fc3_LL = nn.LSTM(input_size=self.config['num_filters'] * 8, hidden_size=256,
                                              batch_first=True, bidirectional=True)
                else:
                    if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                        self.fc3_LL = nn.LSTM(input_size=self.config['num_filters'] * 52, hidden_size=256,
                                              batch_first=True, bidirectional=True)
                    elif self.config["dataset"] == 'pamap2':
                        self.fc3_LL = nn.LSTM(input_size=self.config['num_filters'] * 13, hidden_size=256,
                                              batch_first=True, bidirectional=True)
            elif self.config["aggregate"] == "FC":
                if self.config["network"] == "cnn_imu_tpp":
                    self.fc3_LL = nn.Linear(self.tpp.get_output(), 256)
                else:
                    if self.config["reshape_input"]:
                        if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                            self.fc3_LL = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 15), 256)
                        elif self.config["dataset"] == 'pamap2':
                            self.fc3_LL = nn.Linear(self.config['num_filters'] * int(Wx) * 8, 256)
                    else:
                        if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                            self.fc3_LL = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    52, 256)
                        elif self.config["dataset"] == 'pamap2':
                            self.fc3_LL = nn.Linear(self.config['num_filters'] * int(Wx) * 13, 256)

            # N
            self.conv_N_1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padd)

            self.conv_N_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            self.conv_N_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            self.conv_N_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            if self.config["aggregate"] == "FCN":
                self.fc3_N = nn.Conv2d(in_channels=self.config['num_filters'],
                                       out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["aggregate"] == "LSTM":
                if self.config["reshape_input"]:
                    if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                        self.fc3_N = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 15), hidden_size=256,
                                              batch_first=True, bidirectional=True)
                    elif self.config["dataset"] == 'pamap2':
                        self.fc3_N = nn.LSTM(input_size=self.config['num_filters'] * 6, hidden_size=256,
                                              batch_first=True, bidirectional=True)
                else:
                    if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                        self.fc3_N = nn.LSTM(input_size=self.config['num_filters'] * 45, hidden_size=256,
                                              batch_first=True, bidirectional=True)
                    elif self.config["dataset"] == 'pamap2':
                        self.fc3_N = nn.LSTM(input_size=self.config['num_filters'] * 14, hidden_size=256,
                                              batch_first=True, bidirectional=True)
            elif self.config["aggregate"] == "FC":
                if self.config["network"] == "cnn_imu_tpp":
                    self.fc3_N = nn.Linear(self.tpp.get_output(), 256)
                else:
                    if self.config["reshape_input"]:
                        if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                            self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                   int(self.config['NB_sensor_channels'] / 15), 256)
                        elif self.config["dataset"] == 'pamap2':
                            self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) * 6, 256)
                    else:
                        if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                            self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                   45, 256)
                        elif self.config["dataset"] == 'pamap2':
                            self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) * 14, 256)


            # RA
            self.conv_RA_1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padd)

            self.conv_RA_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            self.conv_RA_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            self.conv_RA_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            if self.config["aggregate"] == "FCN":
                self.fc3_RA = nn.Conv2d(in_channels=self.config['num_filters'],
                                        out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["aggregate"] == "LSTM":
                if self.config["reshape_input"]:
                    if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                        self.fc3_RA = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 15), hidden_size=256,
                                              batch_first=True, bidirectional=True)
                    elif self.config["dataset"] == 'pamap2':
                        self.fc3_RA = nn.LSTM(input_size=self.config['num_filters'] * 10, hidden_size=256,
                                              batch_first=True, bidirectional=True)
                else:
                    if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                        self.fc3_RA = nn.LSTM(input_size=self.config['num_filters'] * 54, hidden_size=256,
                                              batch_first=True, bidirectional=True)
                    elif self.config["dataset"] == 'pamap2':
                        self.fc3_RA = nn.LSTM(input_size=self.config['num_filters'] * 13, hidden_size=256,
                                              batch_first=True, bidirectional=True)
            elif self.config["aggregate"] == "FC":
                if self.config["network"] == "cnn_imu_tpp":
                    self.fc3_RA = nn.Linear(self.tpp.get_output(), 256)
                else:
                    if self.config["reshape_input"]:
                        if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                            self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 15), 256)
                        elif self.config["dataset"] == 'pamap2':
                            self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) * 10, 256)
                    else:
                        if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                            self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    54, 256)
                        elif self.config["dataset"] == 'pamap2':
                            self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) * 13, 256)

            # RL
            self.conv_RL_1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padd)

            self.conv_RL_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            self.conv_RL_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            self.conv_RL_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padd)

            if self.config["aggregate"] == "FCN":
                self.fc3_RL = nn.Conv2d(in_channels=self.config['num_filters'],
                                        out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["aggregate"] == "LSTM":
                if self.config["reshape_input"]:
                    if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                        self.fc3_RL = nn.LSTM(input_size=self.config['num_filters'] *
                                                         int(self.config['NB_sensor_channels'] / 15), hidden_size=256,
                                              batch_first=True, bidirectional=True)
                    elif self.config["dataset"] == 'pamap2':
                        self.fc3_RL = nn.LSTM(input_size=self.config['num_filters'] * 8, hidden_size=256,
                                              batch_first=True, bidirectional=True)
                else:
                    if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                        self.fc3_RL = nn.LSTM(input_size=self.config['num_filters'] * 52, hidden_size=256,
                                              batch_first=True, bidirectional=True)
                    elif self.config["dataset"] == 'pamap2':
                        self.fc3_RL = nn.LSTM(input_size=self.config['num_filters'] * 13, hidden_size=256,
                                              batch_first=True, bidirectional=True)
            elif self.config["aggregate"] == "FC":
                if self.config["network"] == "cnn_imu_tpp":
                    self.fc3_RL = nn.Linear(self.tpp.get_output(), 256)
                else:
                    if self.config["reshape_input"]:
                        if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                            self.fc3_RL = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    int(self.config['NB_sensor_channels'] / 15), 256)
                        elif self.config["dataset"] == 'pamap2':
                            self.fc3_RL = nn.Linear(self.config['num_filters'] * int(Wx) * 8, 256)
                    else:
                        if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                            self.fc3_RL = nn.Linear(self.config['num_filters'] * int(Wx) *
                                                    52, 256)
                        elif self.config["dataset"] == 'pamap2':
                            self.fc3_RL = nn.Linear(self.config['num_filters'] * int(Wx) * 13, 256)


        # MLP
        if self.config["aggregate"] == "FCN":
            if self.config["network"] in ["cnn"]:
                self.fc4 = nn.Conv2d(in_channels=256,
                                     out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["network"] in ["cnn_imu"]:
                self.fc4 = nn.Conv2d(in_channels=256,
                                     out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
        elif self.config["aggregate"] == "FC":
            if self.config["network"] in ["cnn", "cnn_tpp"]:
                self.fc4 = nn.Linear(256, 256)
            elif self.config["network"] in ["cnn_imu", "cnn_imu_tpp"]:
                self.fc4 = nn.Linear(256 * 5, 256)
        elif self.config["aggregate"] == "LSTM":
            if self.config["network"] in ["cnn"]:
                self.fc4 = nn.LSTM(input_size=256 * 2, hidden_size=256, batch_first=True, bidirectional=True)
            if self.config["network"] in ["cnn_imu"]:
                self.fc4 = nn.LSTM(input_size=256 * 10, hidden_size=256, batch_first=True, bidirectional=True)
                # The number of input size is double as one has bidirectional LSTM

        if self.config["aggregate"] == "FCN":
            if self.config['output'] == 'softmax':
                self.fc5 = nn.Conv2d(in_channels=256,
                                     out_channels=self.config['num_classes'], kernel_size=(1, 1), stride=1, padding=0)
            elif self.config['output'] == 'attribute':
                self.fc5 = nn.Conv2d(in_channels=256,
                                     out_channels=self.config['num_attributes'],
                                     kernel_size=(1, 1), stride=1, padding=0)
        elif self.config["aggregate"] == "FC":
            if self.config['output'] == 'softmax':
                self.fc5 = nn.Linear(256, self.config['num_classes'])
            elif self.config['output'] == 'attribute':
                self.fc5 = nn.Linear(256, self.config['num_attributes'])
        elif self.config["aggregate"] == "LSTM":
            if self.config['output'] == 'softmax':
                self.fc5 = nn.Linear(512, self.config['num_classes'])
            elif self.config['output'] == 'attribute':
                self.fc5 = nn.Linear(512, self.config['num_attributes'])
                # The number of input size is double as one has bidirectional LSTM

        if self.config["reshape_input"]:
            self.avgpool = nn.AvgPool2d(kernel_size=[1, int(self.config['NB_sensor_channels'] / 3)])
        else:
            if self.config["network"] == "cnn_imu" and self.config["dataset"] in ["locomotion", "gesture"]:
                self.avgpool = nn.AvgPool2d(kernel_size=[1, 257])
            else:
                self.avgpool = nn.AvgPool2d(kernel_size=[1, self.config['NB_sensor_channels']])

        self.softmax = nn.Softmax(dim=1)
        
        self.sigmoid = nn.Sigmoid()
        
        
        return
    
    
    
    
    
    def forward(self, x):
        '''
        Forwards function, required by torch.

        @param x: batch [batch, 1, Channels, Time], Channels = Sensors * 3 Axis
        @return x: Output of the network, either Softmax or Attribute
        '''

        #fft_output, fft = spectral_pooling(x)
        if self.config["reshape_input"]:
            x = x.permute(0, 2, 1, 3)
            x = x.view(x.size()[0], x.size()[1], int(x.size()[3] / 3), 3)
            x = x.permute(0, 3, 1, 2)

        # Selecting the one ot the two networks, tCNN or tCNN-IMU
        if self.config["network"] == "cnn" or self.config["network"] == "cnn_tpp":
            x = self.tcnn(x)
        elif self.config["network"] == "cnn_imu" or self.config["network"] == "cnn_imu_tpp":
                x_LA, x_LL, x_N, x_RA, x_RL = self.tcnn_imu(x)
                if self.config["aggregate"] in ["FCN"]:
                    x = torch.cat((x_LA, x_LL, x_N, x_RA, x_RL), 3)
                elif self.config["aggregate"] == "FC":
                    x = torch.cat((x_LA, x_LL, x_N, x_RA, x_RL), 1)
                elif self.config["aggregate"] == "LSTM":
                    x = torch.cat((x_LA, x_LL, x_N, x_RA, x_RL), 2)

        # Selecting MLP, either FC or FCN
        if self.config["aggregate"] == "FCN":
            #x = F.dropout(x, training=self.training)
            x = F.dropout2d(x, training=self.training)
            x = F.relu(self.fc4(x))
            #x = F.dropout(x, training=self.training)
            x = F.dropout2d(x, training=self.training)
            x = self.fc5(x)
            #x = F.relu(self.fc5(x))
            x = self.avgpool(x)
            x = x.view(x.size()[0], x.size()[1], x.size()[2])
            x = x.permute(0, 2, 1)
        elif self.config["aggregate"] == "FC":
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc4(x))
            x = F.dropout(x, training=self.training)
            x = self.fc5(x)
        elif self.config["aggregate"] == "LSTM":
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc4(x)[0])
            x = F.dropout(x, training=self.training)
            #x = x[:, -1]
            x = self.fc5(x)

        if self.training:
            if self.config['output'] == 'attribute':
                x = self.sigmoid(x)
        elif not self.training:
            if self.config['output'] == 'softmax':
                if self.config["aggregate"] == "FCN":
                    x = x.reshape(-1, x.size()[2])
                if self.config["aggregate"] == "LSTM":
                    x = x.reshape(-1, x.size()[2])
                x = self.softmax(x)
            elif self.config['output'] == 'attribute':
                if self.config["aggregate"] == "FCN":
                    x = x.reshape(-1, x.size()[2])
                if self.config["aggregate"] == "LSTM":
                    x = x.reshape(-1, x.size()[2])
                x = self.sigmoid(x)

        return x
    
    
    
    def init_weights(self):
        self.apply(Network._init_weights_orthonormal)
        return
    
    
    
    @staticmethod
    def _init_weights_orthonormal(m):
        if isinstance(m, nn.Conv2d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias.data, 0)
        if isinstance(m, nn.Linear):          
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias.data, 0)
        
        return




    def size_feature_map(self, Wx, Hx, F, P, S, type_layer = 'conv'):
        '''
        Computing size of feature map after convolution or pooling

        @param Wx: Width input
        @param Hx: Height input
        @param F: Filter size
        @param P: Padding
        @param S: Stride
        @param type_layer: conv or pool
        @return Wy: Width output
        @return Hy: Height output
        '''

        if self.config["aggregate"] in ["FCN", "LSTM"]:
            Pw = P[0]
            Ph = P[1]
        elif self.config["aggregate"] == "FC":
            Pw = P
            Ph = P
        elif self.config["aggregate"] == "LSTM2":
            Pw = P
            Ph = P

        if type_layer == 'conv':
            Wy = 1 + (Wx - F[0] + 2 * Pw) / S[0]
            Hy = 1 + (Hx - F[1] + 2 * Ph) / S[1]

        elif type_layer == 'pool':
            Wy = 1 + (Wx - F[0]) / S[0]
            Hy = 1 + (Hx - F[1]) / S[1]

        return Wy, Hy


    def tcnn(self, x):
        '''
        tCNN network

        @param x: input sequence
        @return x: Prediction of the network
        '''
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        if self.config["pooling"] in [1, 2]:
            x = self.spectral_pooling(x, pooling_number=0)
        elif self.config["pooling"] in [3, 4]:
            x = F.max_pool2d(x, (2, 1))
        # fft_output = x.clone().detach()

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))

        if self.config["pooling"] == 2:
            x = self.spectral_pooling(x, pooling_number=1)
        elif self.config["pooling"] == 4:
            x = F.max_pool2d(x, (2, 1))
        # fft_output = x.clone().detach()

        if self.config["network"] == "cnn_tpp":
            x = self.tpp.tpp(x)
            x = F.relu(self.fc3(x))
        else:
            if self.config["aggregate"] == "FCN":
                x = F.relu(self.fc3(x))
            elif self.config["aggregate"] == "FC":
                # view is reshape
                x = x.reshape((-1, x.size()[1] * x.size()[2] * x.size()[3]))
                x = F.relu(self.fc3(x))
            elif self.config["aggregate"] == "LSTM":
                x = x.permute(0, 2, 1, 3)
                x = x.reshape((x.size()[0], x.size()[1], x.size()[2] * x.size()[3]))
                x = F.relu(self.fc3(x)[0])
        return x



    def tcnn_imu(self, x):
        '''
        tCNN-IMU network
        The parameters will adapt according to the dataset, reshape and output type

        x_LA, x_LL, x_N, x_RA, x_RL

        @param x: input sequence
        @return x_LA: Features from left arm
        @return x_LL: Features from left leg
        @return x_N: Features from Neck or Torso
        @return x_RA: Features from Right Arm
        @return x_RL: Features from Right Leg
        '''
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
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LA]))
            elif self.config["dataset"] == 'pamap2':
                idx_LA = np.arange(1, 14)
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LA]))

        x_LA = F.relu(self.conv_LA_1_2(x_LA))
        if self.config["pooling"] in [1, 2]:
            x_LA = self.spectral_pooling(x_LA, pooling_number=0)
        elif self.config["pooling"] in [3, 4]:
            x_LA = F.max_pool2d(x_LA, (2, 1))
        x_LA = F.relu(self.conv_LA_2_1(x_LA))
        x_LA = F.relu(self.conv_LA_2_2(x_LA))
        if self.config["pooling"] == 2:
            x_LA = self.spectral_pooling(x_LA, pooling_number=1)
        elif self.config["pooling"] == 4:
            x_LA = F.max_pool2d(x_LA, (2, 1))

        if self.config["network"] == "cnn_imu_tpp":
            x_LA = self.tpp.tpp(x_LA)
            x_LA = F.relu(self.fc3_LA(x_LA))
        else:
            if self.config["aggregate"] == "FCN":
                x_LA = F.relu(self.fc3_LA(x_LA))
            elif self.config["aggregate"] == "FC":
                # view is reshape
                x_LA = x_LA.reshape(-1, x_LA.size()[1] * x_LA.size()[2] * x_LA.size()[3])
                x_LA = F.relu(self.fc3_LA(x_LA))
            elif self.config["aggregate"] == "LSTM":
                x_LA = x_LA.permute(0, 2, 1, 3)
                x_LA = x_LA.reshape((x_LA.size()[0], x_LA.size()[1], x_LA.size()[2] * x_LA.size()[3]))
                x_LA = F.relu(self.fc3_LA(x_LA)[0])


        # LL
        if self.config["reshape_input"]:
            if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                idx_LL = np.arange(0, 36)
                idx_LL = np.concatenate([idx_LL, np.arange(81, 97)])
                x_LL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LL]))
            elif self.config["dataset"] == 'pamap2':
                idx_LL = np.arange(27, 40)
                x_LL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LL]))
        else:
            if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                idx_LL = np.arange(0, 36)
                idx_LL = np.concatenate([idx_LL, np.arange(81, 97)])
                x_LL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LL]))
            elif self.config["dataset"] == 'pamap2':
                idx_LL = np.arange(27, 40)
                x_LL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LL]))

        x_LL = F.relu(self.conv_LL_1_2(x_LL))
        if self.config["pooling"] in [1, 2]:
            x_LL = self.spectral_pooling(x_LL, pooling_number=0)
        elif self.config["pooling"] in [3, 4]:
            x_LL = F.max_pool2d(x_LL, (2, 1))
        x_LL = F.relu(self.conv_LL_2_1(x_LL))
        x_LL = F.relu(self.conv_LL_2_2(x_LL))
        if self.config["pooling"] == 2:
            x_LL = self.spectral_pooling(x_LL, pooling_number=1)
        elif self.config["pooling"] == 4:
            x_LL = F.max_pool2d(x_LL, (2, 1))

        if self.config["network"] == "cnn_imu_tpp":
            x_LL = self.tpp.tpp(x_LL)
            x_LL = F.relu(self.fc3_LL(x_LL))
        else:
            if self.config["aggregate"] == "FCN":
                x_LL = F.relu(self.fc3_LL(x_LL))
            elif self.config["aggregate"] == "FC":
                # view is reshape
                x_LL = x_LL.reshape(-1, x_LL.size()[1] * x_LL.size()[2] * x_LL.size()[3])
                x_LL = F.relu(self.fc3_LL(x_LL))
            elif self.config["aggregate"] == "LSTM":
                x_LL = x_LL.permute(0, 2, 1, 3)
                x_LL = x_LL.reshape((x_LL.size()[0], x_LL.size()[1], x_LL.size()[2] * x_LL.size()[3]))
                x_LL = F.relu(self.fc3_LL(x_LL)[0])

        # N
        if self.config["reshape_input"]:
            if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                idx_N = np.arange(0, 36)
                idx_N = np.concatenate([idx_N, np.arange(36, 45)])
                x_N = F.relu(self.conv_LA_1_1(x[:, :, :, idx_N]))
            elif self.config["dataset"] == 'pamap2':
                idx_N = np.arange(0, 1)
                idx_N = np.concatenate([idx_N, np.arange(14, 27)])
                x_N = F.relu(self.conv_LA_1_1(x[:, :, :, idx_N]))
        else:
            if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                idx_N = np.arange(0, 36)
                idx_N = np.concatenate([idx_N, np.arange(36, 45)])
                x_N = F.relu(self.conv_LA_1_1(x[:, :, :, idx_N]))
            elif self.config["dataset"] == 'pamap2':
                idx_N = np.arange(0, 1)
                idx_N = np.concatenate([idx_N, np.arange(14, 27)])
                x_N = F.relu(self.conv_LA_1_1(x[:, :, :, idx_N]))
        x_N = F.relu(self.conv_N_1_2(x_N))
        if self.config["pooling"] == 1 or self.config["pooling"] == 2:
            x_N = self.spectral_pooling(x_N, pooling_number=0)
        elif self.config["pooling"] in [3, 4]:
            x_N = F.max_pool2d(x_N, (2, 1))
        x_N = F.relu(self.conv_N_2_1(x_N))
        x_N = F.relu(self.conv_N_2_2(x_N))
        if self.config["pooling"] == 2:
            x_N = self.spectral_pooling(x_N, pooling_number=1)
        elif self.config["pooling"] == 4:
            x_N = F.max_pool2d(x_N, (2, 1))

        if self.config["network"] == "cnn_imu_tpp":
            x_N = self.tpp.tpp(x_N)
            x_N = F.relu(self.fc3_N(x_N))
        else:
            if self.config["aggregate"] == "FCN":
                x_N = F.relu(self.fc3_N(x_N))
            elif self.config["aggregate"] == "FC":
                # view is reshape
                x_N = x_N.reshape(-1, x_N.size()[1] * x_N.size()[2] * x_N.size()[3])
                x_N = F.relu(self.fc3_N(x_N))
            elif self.config["aggregate"] == "LSTM":
                x_N = x_N.permute(0, 2, 1, 3)
                x_N = x_N.reshape((x_N.size()[0], x_N.size()[1], x_N.size()[2] * x_N.size()[3]))
                x_N = F.relu(self.fc3_N(x_N)[0])

        # RA
        if self.config["reshape_input"]:
            if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                idx_RA = np.arange(0, 36)
                idx_RA = np.concatenate([idx_RA, np.arange(54, 63)])
                idx_RA = np.concatenate([idx_RA, np.arange(63, 72)])
                x_RA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RA]))
            elif self.config["dataset"] == 'pamap2':
                idx_RA = np.arange(1, 14)
                x_RA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RA]))
        else:
            if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                idx_RA = np.arange(0, 36)
                idx_RA = np.concatenate([idx_RA, np.arange(54, 63)])
                idx_RA = np.concatenate([idx_RA, np.arange(63, 72)])
                x_RA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RA]))
            elif self.config["dataset"] == 'pamap2':
                idx_RA = np.arange(1, 14)
                x_RA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RA]))

        x_RA = F.relu(self.conv_RA_1_2(x_RA))
        if self.config["pooling"] in [1, 2]:
            x_RA = self.spectral_pooling(x_RA, pooling_number=0)
        elif self.config["pooling"] in [3, 4]:
            x_RA = F.max_pool2d(x_RA, (2, 1))
        x_RA = F.relu(self.conv_RA_2_1(x_RA))
        x_RA = F.relu(self.conv_RA_2_2(x_RA))
        if self.config["pooling"] == 2:
            x_RA = self.spectral_pooling(x_RA, pooling_number=1)
        elif self.config["pooling"] == 4:
            x_RA = F.max_pool2d(x_RA, (2, 1))

        if self.config["network"] == "cnn_imu_tpp":
            x_RA = self.tpp.tpp(x_RA)
            x_RA = F.relu(self.fc3_RA(x_RA))
        else:
            if self.config["aggregate"] == "FCN":
                x_RA = F.relu(self.fc3_RA(x_RA))
            elif self.config["aggregate"] == "FC":
                # view is reshape
                x_RA = x_RA.reshape(-1, x_RA.size()[1] * x_RA.size()[2] * x_RA.size()[3])
                x_RA = F.relu(self.fc3_RA(x_RA))
            elif self.config["aggregate"] == "LSTM":
                x_RA = x_RA.permute(0, 2, 1, 3)
                x_RA = x_RA.reshape((x_RA.size()[0], x_RA.size()[1], x_RA.size()[2] * x_RA.size()[3]))
                x_RA = F.relu(self.fc3_RA(x_RA)[0])

        # RL
        if self.config["reshape_input"]:
            if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                idx_RL = np.arange(0, 36)
                idx_RL = np.concatenate([idx_RL, np.arange(81, 97)])
                x_RL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RL]))
            elif self.config["dataset"] == 'pamap2':
                idx_RL = np.arange(27, 40)
                x_RL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RL]))
        else:
            if self.config["dataset"] == 'locomotion' or self.config["dataset"] == 'gesture':
                idx_RL = np.arange(0, 36)
                idx_RL = np.concatenate([idx_RL, np.arange(81, 97)])
                x_RL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RL]))
            elif self.config["dataset"] == 'pamap2':
                idx_RL = np.arange(27, 40)
                x_RL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RL]))

        x_RL = F.relu(self.conv_RL_1_2(x_RL))
        if self.config["pooling"] in [1, 2]:
            x_RL = self.spectral_pooling(x_RL, pooling_number=0)
        elif self.config["pooling"] in [3, 4]:
            x_RL = F.max_pool2d(x_RL, (2, 1))
        x_RL = F.relu(self.conv_RL_2_1(x_RL))
        x_RL = F.relu(self.conv_RL_2_2(x_RL))
        if self.config["pooling"] == 2:
            x_RL = self.spectral_pooling(x_RL, pooling_number=1)
        elif self.config["pooling"] == 4:
            x_RL = F.max_pool2d(x_RL, (2, 1))

        if self.config["network"] == "cnn_imu_tpp":
            x_RL = self.tpp.tpp(x_RL)
            x_RL = F.relu(self.fc3_RL(x_RL))
        else:
            if self.config["aggregate"] == "FCN":
                x_RL = F.relu(self.fc3_RL(x_RL))
            elif self.config["aggregate"] == "FC":
                # view is reshape
                x_RL = x_RL.reshape(-1, x_RL.size()[1] * x_RL.size()[2] * x_RL.size()[3])
                x_RL = F.relu(self.fc3_RL(x_RL))
            elif self.config["aggregate"] == "LSTM":
                x_RL = x_RL.permute(0, 2, 1, 3)
                x_RL = x_RL.reshape((x_RL.size()[0], x_RL.size()[1], x_RL.size()[2] * x_RL.size()[3]))
                x_RL = F.relu(self.fc3_RL(x_RL)[0])

        return x_LA, x_LL, x_N, x_RA, x_RL


    def spectral_pooling(self, x, pooling_number = 0):
        '''
        Carry out a spectral pooling.
        torch.rfft(x, signal_ndim, normalized, onesided)
        signal_ndim takes into account the signal_ndim dimensions stranting from the last one
        onesided if True, outputs only the positives frequencies, under the nyquist frequency

        @param x: input sequence
        @return x: output of spectral pooling
        '''
        # xpool = F.max_pool2d(x, (2, 1))

        x = x.permute(0, 1, 3, 2)

        # plt.figure()
        # f, axarr = plt.subplots(5, 1)

        # x_plt = x[0, 0].to("cpu", torch.double).detach()
        # axarr[0].plot(x_plt[0], label='input')

        #fft = torch.rfft(x, signal_ndim=1, normalized=True, onesided=True)
        fft = torch.fft.rfft(x, norm="forward")
        # fft2 = torch.rfft(x, signal_ndim=1, normalized=False, onesided=False)

        # fft_plt = fft[0, 0].to("cpu", torch.double).detach()
        # fft_plt = torch.norm(fft_plt, dim=2)
        # axarr[1].plot(fft_plt[0], 'o', label='fft')

        x = fft[:, :, :, :int(math.ceil(fft.shape[3] / 2))]

        # fftx_plt = x[0, 0].to("cpu", torch.double).detach()
        # fftx_plt = torch.norm(fftx_plt, dim=2)
        # axarr[2].plot(fftx_plt[0], 'o', label='fft')

        # x = torch.irfft(x, signal_ndim=1, normalized=True, onesided=True)
        x = torch.fft.irfft(x, norm="forward")

        x = x[:, :, :, :self.pooling_Wx[pooling_number]]

        # x_plt = x[0, 0].to("cpu", torch.double).detach()
        # axarr[3].plot(x_plt[0], label='input')

        x = x.permute(0, 1, 3, 2)

        # fft2_plt = fft2[0, 0].to("cpu", torch.double).detach()
        # fft2_plt = torch.norm(fft2_plt, dim=2)
        # print(fft2_plt.size(), 'max: {}'.format(torch.max(fft2_plt)), 'min: {}'.format(torch.min(fft2_plt)))
        # axarr[4].plot(fft2_plt[0], 'o', label='fft')

        # xpool = xpool.permute(0, 1, 3, 2)
        # x_plt = xpool[0, 0].to("cpu", torch.double).detach()
        # axarr[3].plot(x_plt[0], label='input')

        # plt.waitforbuttonpress(0)
        # plt.close()


        return x

