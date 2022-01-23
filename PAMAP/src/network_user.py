'''
Created on Mar 5, 2019

@author: fmoya

Old network_user with caffe/theano implementations

'''

from __future__ import print_function
import os
import logging
import numpy as np
import time
import math

import pickle

import torch, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sliding_window import sliding_window

from metrics import Metrics

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.collections import PolyCollection

from network import Network
from opportunity import Opportunity
from pamap2 import Pamap2
from orderpicking import OrderPicking




class Network_User(object):
    '''
    classdocs
    '''

    def __init__(self, config, exp=None):
        '''
        Constructor
        '''

        logging.info('        Network_User: Constructor')

        self.config = config
        self.device = torch.device("cuda:{}".format(self.config["GPU"]) if torch.cuda.is_available() else "cpu")
        self.network_obj = Network(self.config)

        if self.config['dataset'] == 'locomotion':
            self.attrs = self.reader_att_rep("atts_per_class_locomotion.txt")
            self.attr_representation = self.reader_att_rep("atts_per_class_locomotion.txt")
        elif self.config['dataset'] == 'gesture':
            self.attrs = self.reader_att_rep("atts_per_class_gesture.txt")
            self.attr_representation = self.reader_att_rep("atts_per_class_gesture.txt")
        elif self.config['dataset'] == 'pamap2':
            #self.attrs = self.reader_att_rep("atts_per_class_pamap2.txt")
            self.attrs = self.reader_att_rep("pamap_attr.txt")
            self.attr_representation = self.reader_att_rep("pamap_attr.txt")
        else:
            self.attrs = None
            self.attr_representation = None

        logging.info('        Network_User:     Creating Dataloader---->')

        if self.config['dataset'] == 'locomotion' or self.config['dataset'] == 'gesture':
            self.harwindows_train = Opportunity(self.config, partition_modus='train')
            self.harwindows_val = Opportunity(self.config, partition_modus='val')
            self.harwindows_test = Opportunity(self.config, partition_modus='test')
        elif self.config['dataset'] == 'pamap2':
            self.harwindows_train = Pamap2(self.config, partition_modus='train')
            self.harwindows_val = Pamap2(self.config, partition_modus='val')
            self.harwindows_test = Pamap2(self.config, partition_modus='test')
        elif self.config['dataset'] == 'orderpicking':
            self.harwindows_train = OrderPicking(self.config, partition_modus='train')
            self.harwindows_val = OrderPicking(self.config, partition_modus='test')
            self.harwindows_test = OrderPicking(self.config, partition_modus='test')

        self.dataLoader_train = DataLoader(self.harwindows_train, batch_size=self.config['batch_size_train'], shuffle=True)
        self.dataLoader_val = DataLoader(self.harwindows_val, batch_size=self.config['batch_size_val'])
        self.dataLoader_test = DataLoader(self.harwindows_test, batch_size=self.config['batch_size_val'])

        self.normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.001]))

        self.exp = exp

        return


    ##################################################
    ###################  reader_att_rep  #############
    ##################################################

    def reader_att_rep(self, path: str) -> np.array:
        '''
        gets attribute representation from txt file.

        returns a numpy array

        @param path: path to file
        @param att_rep: Numpy matrix with the attribute representation
        '''

        att_rep = np.loadtxt(path, delimiter=',', skiprows=0)
        return att_rep

    ##################################################
    ################  load_dataset  ##################
    ##################################################

    def load_dataset(self, filename):
        logging.info('        Network_User: Loading Dataset from file {}'.format(filename))

        try:
            f = open(filename, 'rb')
            data = pickle.load(f)
            f.close()
        except:
            logging.error("No such file ro directory")

        X_train, y_train = data[0]
        X_val, y_val = data[1]
        X_test, y_test = data[2]

        logging.info(
            '        Network_User: Train shape {0}, Train shape {1}, Train shape {2}'.format(X_train.shape, X_val.shape,
                                                                                             X_test.shape))

        # The targets are casted to float 32 for GPU compatibility.
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # The targets are casted to int8 for GPU compatibility.
        y_train = y_train.astype(np.uint8)
        y_val = y_val.astype(np.uint8)
        y_test = y_test.astype(np.uint8)

        return X_train, y_train, X_val, y_val, X_test, y_test

    ##################################################    
    #############  opp_sliding_window  ###############
    ##################################################

    def opp_sliding_window(self, data_x, data_y):
        ws = self.config['sliding_window_length']
        ss = self.config['sliding_window_step']

        logging.info('        Network_User: Sliding window with ws {} and ss {}'.format(ws, ss))

        # Segmenting the data with labels taken from the end of the window
        data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
        if self.config['label_pos'] == 'end':
            data_y_labels = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
        elif self.config['label_pos'] == 'middle':
            # Segmenting the data with labels from the middle of the window
            data_y_labels = np.asarray([[i[i.shape[0] // 2]] for i in sliding_window(data_y, ws, ss)])
        elif self.config['label_pos'] == 'mode':
            data_y_labels = []
            for sw in sliding_window(data_y, ws, ss):
                count_l = np.bincount(sw, minlength=self.num_classes)
                idy = np.argmax(count_l)
                data_y_labels.append(idy)
            data_y_labels = np.asarray(data_y_labels)

        # Labels of each sample per window
        data_y_all = np.asarray([i[:] for i in sliding_window(data_y, ws, ss)])

        logging.info('        Network_User: Sequences are segmented')

        return data_x.astype(np.float32), data_y_labels.reshape(len(data_y_labels)).astype(np.uint8), data_y_all.astype(
            np.uint8)

    ##################################################    
    ###############  create_batches  #################
    ##################################################

    def create_batches(self, data, batch_size=1):

        logging.info('        Network_User: Preparing data with batch size {}'.format(batch_size))
        data_batches = []
        batches = np.arange(0, data.shape[0], batch_size)

        for idx in range(batches.shape[0] - 1):
            batch = []
            for data_in_batch in data[batches[idx]: batches[idx + 1]]:
                channel = []
                channel.append(data_in_batch.astype(np.float32))
                batch.append(channel)
            data_batches.append(batch)

        data_batches = np.array(data_batches)

        return data_batches

    ##################################################    
    ################  random_data  ###################
    ################################################## 

    def random_data(self, data, label, y_data=None):
        logging.info('        Network_User: Randomizing data')

        if data.shape[0] != label.shape[0]:
            logging.info('        Network_User: Random: Data and label dont have the same number of samples')
            raise RuntimeError('Random: Data and label havent the same number of samples')

        if os.path.isfile(self.config['folder_exp'] + 'random_train_order.pkl'):
            logging.info("        Network_User: Getting random order")

            file2idx = pickle.load(open(self.config['folder_exp'] + 'random_train_order.pkl', 'rb'))
            idx = file2idx["idx"]

        else:
            idx = np.arange(data.shape[0])
            np.random.shuffle(idx)

            idx2file = {"idx": idx}
            f = open(self.config['folder_exp'] + 'random_train_order.pkl', 'wb')
            pickle.dump(idx2file, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        data_s = data[idx]
        label_s = label[idx]

        if y_data is not None:
            y_data_s = y_data[idx]
        else:
            y_data_s = y_data

        return data_s, label_s, y_data_s

    ##################################################    
    ###############  prepare_data  ###################
    ##################################################    

    def prepare_data(self, values, labels, if_val=False, batch_size=1, y_data=None):
        logging.info('        Network_User: Preparing data')

        if if_val == False:
            logging.info('        Network_User: Preparing Train data')
            train_vals_fl, train_labels_fl, y_data_fl = self.random_data(values, labels, y_data=y_data)

        else:
            logging.info('        Network_User: Preparing Val data')
            train_vals_fl = values
            train_labels_fl = labels
            y_data_fl = y_data

        logging.info('        Network_User: Creating Batches')

        v_b = np.array(self.create_batches(np.array(train_vals_fl), batch_size=batch_size))
        l_b = np.array(self.create_batches(np.array(train_labels_fl), batch_size=batch_size))

        if y_data is not None:
            y_data_b = np.array(self.create_batches(np.array(y_data_fl), batch_size=batch_size))
        else:
            y_data_b = None

        return v_b.astype(np.float32), l_b.astype(np.float32), y_data_b




    ##################################################    
    ############  set save network  ##################
    ##################################################

    def save_network(self, itera, name_net='best_network'):

        logging.info('        Network_User: Saving network---->')
        torch.save({'state_dict': self.network_obj.state_dict()}, self.config['folder_exp'] + name_net + '.pt')

        return

    ##################################################
    ############  set save network  ##################
    ##################################################

    def restart_network(self):

        logging.info('        Network_User: Restarting network---->')
        self.network_obj = Network(self.config)

        return

    ##################################################
    #################  set attrs  ######################
    ##################################################

    def set_attrs(self, attrs):

        logging.info('        Network_User: Setting attributes---->')
        self.attrs = np.copy(attrs)

        return

    ##################################################
    ###################  Plot  ######################
    ##################################################

    def plot(self, fig, axis_list, plot_list, metrics_list, activaciones, tgt, pred):

        logging.info('        Network_User:    Plotting')
        if self.config['plotting']:
            # Plot

            for an, act in enumerate(activaciones):
                X = np.arange(0, act.shape[1])
                Y = np.arange(0, act.shape[0])
                X, Y = np.meshgrid(X, Y)

                axis_list[an * 2].plot_surface(X, Y, act, cmap=cm.coolwarm, linewidth=1, antialiased=False)

                axis_list[an * 2].set_title('Target {} and Pred {}'.format(tgt, pred))
                axis_list[an * 2].set_xlim3d(X.min(), X.max())
                axis_list[an * 2].set_xlabel('Sensor')
                axis_list[an * 2].set_ylim3d(Y.min(), Y.max())
                axis_list[an * 2].set_ylabel('Time')
                axis_list[an * 2].set_zlim3d(act.min(), act.max())
                axis_list[an * 2].set_zlabel('Measurement')

            for pl in range(len(metrics_list)):
                plot_list[pl].set_ydata(metrics_list[pl])
                plot_list[pl].set_xdata(range(len(metrics_list[pl])))

            axis_list[1].relim()
            axis_list[1].autoscale_view()
            axis_list[1].legend(loc='best')

            axis_list[3].relim()
            axis_list[3].autoscale_view()
            axis_list[3].legend(loc='best')

            axis_list[5].relim()
            axis_list[5].autoscale_view()
            axis_list[5].legend(loc='best')

            axis_list[7].relim()
            axis_list[7].autoscale_view()
            axis_list[7].legend(loc='best')

            fig.canvas.draw()
            plt.savefig(self.config['folder_exp'] + 'training.png')
            # plt.show()
            plt.pause(2.0)
            axis_list[0].cla()
            axis_list[2].cla()
            axis_list[4].cla()
            axis_list[6].cla()
            axis_list[8].cla()

        return



    ################################################################
    ################  setting_samples_from_windows  ##############
    ################################################################
    def setting_samples_from_windows(self, targets, predictions_test, targets_files = None):
        logging.info("        Network_User:            Segmentation:    ")

        size_samples = self.config["sliding_window_step"] * (targets.size(0) - 1) + \
                       self.config["sliding_window_length"]
        sample_targets = torch.zeros((self.config["num_classes"], size_samples)).to(self.device, dtype=torch.long)
        sample_predictions = torch.zeros((self.config["num_classes"],
                                          size_samples)).to(self.device, dtype=torch.long)
        for ws in range(targets.size(0)):
            window_samples = targets[ws]
            window_samples = nn.functional.one_hot(window_samples.type(torch.LongTensor),
                                                   num_classes=self.config["num_classes"])
            window_samples = torch.transpose(window_samples, 0, 1).to(self.device, dtype=torch.long)
            sample_targets[:, self.config["sliding_window_step"] *
                              ws: (self.config["sliding_window_step"] * ws) +
                                  self.config["sliding_window_length"]] += window_samples

            if self.config["aggregate"] in ["FCN", "LSTM"]:
                window_samples = predictions_test[ws]
            else:
                window_samples = torch.ones((targets[ws].size())).to(self.device, dtype=torch.long)
                window_samples = window_samples * predictions_test[ws]
            window_samples = nn.functional.one_hot(window_samples.type(torch.LongTensor),
                                                   num_classes=self.config["num_classes"])
            window_samples = torch.transpose(window_samples, 0, 1).to(self.device, dtype=torch.long)
            sample_predictions[:, self.config["sliding_window_step"] *
                                   ws: (self.config["sliding_window_step"] * ws) +
                                       self.config["sliding_window_length"]] += window_samples

        sample_targets_single = torch.argmax(sample_targets.type(dtype=torch.long), axis=0)

        sample_predictions_single = torch.argmax(sample_predictions.type(dtype=torch.long), axis=0)


        logging.info("        Network_User:            Segmentation:    "
                     "size of sequence labels {}".format(sample_targets_single.size()))

        return sample_targets_single, sample_predictions_single


    ##################################################    
    ################  load_weights  ##################
    ##################################################
    
    def load_weights(self, network):
        model_dict = network.state_dict()
        # 1. filter out unnecessary keys
        logging.info('        Network_User:        Loading Weights')

        #print(torch.load(self.config['folder_exp_base_fine_tuning'] + 'network.pt')['state_dict'])

        pretrained_dict = torch.load(self.config['folder_exp_base_fine_tuning'] + 'network.pt')['state_dict']
        logging.info('        Network_User:        Pretrained model loaded')

        #for k, v in pretrained_dict.items():
        #    print(k)

        if self.config["network"] == 'cnn':
            list_layers = ['conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias',
                           'conv2_1.weight', 'conv2_1.bias', 'conv2_2.weight', 'conv2_2.bias']
        elif self.config["network"] == 'cnn_tpp':
            list_layers = ['conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias',
                           'conv2_1.weight', 'conv2_1.bias', 'conv2_2.weight', 'conv2_2.bias',
                           'fc3.weight', 'fc3.bias', 'fc4.weight', 'fc4.bias']
        elif self.config["network"] == 'cnn_imu':
            list_layers = ['conv_LA_1_1.weight', 'conv_LA_1_1.bias', 'conv_LA_1_2.weight', 'conv_LA_1_2.bias',
                           'conv_LA_2_1.weight', 'conv_LA_2_1.bias', 'conv_LA_2_2.weight', 'conv_LA_2_2.bias',
                           'conv_LL_1_1.weight', 'conv_LL_1_1.bias', 'conv_LL_1_2.weight', 'conv_LL_1_2.bias',
                           'conv_LL_2_1.weight', 'conv_LL_2_1.bias', 'conv_LL_2_2.weight', 'conv_LL_2_2.bias',
                           'conv_N_1_1.weight', 'conv_N_1_1.bias', 'conv_N_1_2.weight', 'conv_N_1_2.bias',
                           'conv_N_2_1.weight', 'conv_N_2_1.bias', 'conv_N_2_2.weight', 'conv_N_2_2.bias',
                           'conv_RA_1_1.weight', 'conv_RA_1_1.bias', 'conv_RA_1_2.weight', 'conv_RA_1_2.bias',
                           'conv_RA_2_1.weight', 'conv_RA_2_1.bias', 'conv_RA_2_2.weight', 'conv_RA_2_2.bias',
                           'conv_RL_1_1.weight', 'conv_RL_1_1.bias', 'conv_RL_1_2.weight',  'conv_RL_1_2.bias',
                           'conv_RL_2_1.weight', 'conv_RL_2_1.bias', 'conv_RL_2_2.weight', 'conv_RL_2_2.bias']
        elif self.config["network"] == 'cnn_imu_tpp':
            list_layers = ['conv_LA_1_1.weight', 'conv_LA_1_1.bias', 'conv_LA_1_2.weight', 'conv_LA_1_2.bias',
                           'conv_LA_2_1.weight', 'conv_LA_2_1.bias', 'conv_LA_2_2.weight', 'conv_LA_2_2.bias',
                           'conv_LL_1_1.weight', 'conv_LL_1_1.bias', 'conv_LL_1_2.weight', 'conv_LL_1_2.bias',
                           'conv_LL_2_1.weight', 'conv_LL_2_1.bias', 'conv_LL_2_2.weight', 'conv_LL_2_2.bias',
                           'conv_N_1_1.weight', 'conv_N_1_1.bias', 'conv_N_1_2.weight', 'conv_N_1_2.bias',
                           'conv_N_2_1.weight', 'conv_N_2_1.bias', 'conv_N_2_2.weight', 'conv_N_2_2.bias',
                           'conv_RA_1_1.weight', 'conv_RA_1_1.bias', 'conv_RA_1_2.weight', 'conv_RA_1_2.bias',
                           'conv_RA_2_1.weight', 'conv_RA_2_1.bias', 'conv_RA_2_2.weight', 'conv_RA_2_2.bias',
                           'conv_RL_1_1.weight', 'conv_RL_1_1.bias', 'conv_RL_1_2.weight',  'conv_RL_1_2.bias',
                           'conv_RL_2_1.weight', 'conv_RL_2_1.bias', 'conv_RL_2_2.weight', 'conv_RL_2_2.bias',
                           'fc3_LA.weight', 'fc3_LA.bias',
                           'fc3_N.weight', 'fc3_N.bias',
                           'fc3_RA.weight', 'fc3_RA.bias']
                           #'fc3_RL.weight', 'fc3_RL.bias',
                           #'fc3_LL.weight', 'fc3_LL.bias']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in list_layers}
        #print(pretrained_dict)

        logging.info('        Network_User:        Pretrained layers selected')
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        logging.info('        Network_User:        Pretrained layers selected')
        # 3. load the new state dict
        network.load_state_dict(model_dict)
        logging.info('        Network_User:        Weights loaded')

        return network



    ##################################################
    ############  set_required_grad  #################
    ##################################################

    def set_required_grad(self, network):
        model_dict = network.state_dict()
        # 1. filter out unnecessary keys
        logging.info('        Network_User:        Setting Required_grad to Weights')

        if self.config["network"] == 'cnn':
            list_layers = ['conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias',
                           'conv2_1.weight', 'conv2_1.bias', 'conv2_2.weight', 'conv2_2.bias']
        elif self.config["network"] == 'cnn_tpp':
            list_layers = ['conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias',
                           'conv2_1.weight', 'conv2_1.bias', 'conv2_2.weight', 'conv2_2.bias',
                           'fc3.weight', 'fc3.bias', 'fc4.weight', 'fc4.bias']
        elif self.config["network"] == 'cnn_imu':
            list_layers = ['conv_LA_1_1.weight', 'conv_LA_1_1.bias', 'conv_LA_1_2.weight', 'conv_LA_1_2.bias',
                           'conv_LA_2_1.weight', 'conv_LA_2_1.bias', 'conv_LA_2_2.weight', 'conv_LA_2_2.bias',
                           'conv_LL_1_1.weight', 'conv_LL_1_1.bias', 'conv_LL_1_2.weight', 'conv_LL_1_2.bias',
                           'conv_LL_2_1.weight', 'conv_LL_2_1.bias', 'conv_LL_2_2.weight', 'conv_LL_2_2.bias',
                           'conv_N_1_1.weight', 'conv_N_1_1.bias', 'conv_N_1_2.weight', 'conv_N_1_2.bias',
                           'conv_N_2_1.weight', 'conv_N_2_1.bias', 'conv_N_2_2.weight', 'conv_N_2_2.bias',
                           'conv_RA_1_1.weight', 'conv_RA_1_1.bias', 'conv_RA_1_2.weight', 'conv_RA_1_2.bias',
                           'conv_RA_2_1.weight', 'conv_RA_2_1.bias', 'conv_RA_2_2.weight', 'conv_RA_2_2.bias',
                           'conv_RL_1_1.weight', 'conv_RL_1_1.bias', 'conv_RL_1_2.weight', 'conv_RL_1_2.bias',
                           'conv_RL_2_1.weight', 'conv_RL_2_1.bias', 'conv_RL_2_2.weight', 'conv_RL_2_2.bias']
        elif self.config["network"] == 'cnn_imu_tpp':
            list_layers = ['conv_LA_1_1.weight', 'conv_LA_1_1.bias', 'conv_LA_1_2.weight', 'conv_LA_1_2.bias',
                           'conv_LA_2_1.weight', 'conv_LA_2_1.bias', 'conv_LA_2_2.weight', 'conv_LA_2_2.bias',
                           'conv_LL_1_1.weight', 'conv_LL_1_1.bias', 'conv_LL_1_2.weight', 'conv_LL_1_2.bias',
                           'conv_LL_2_1.weight', 'conv_LL_2_1.bias', 'conv_LL_2_2.weight', 'conv_LL_2_2.bias',
                           'conv_N_1_1.weight', 'conv_N_1_1.bias', 'conv_N_1_2.weight', 'conv_N_1_2.bias',
                           'conv_N_2_1.weight', 'conv_N_2_1.bias', 'conv_N_2_2.weight', 'conv_N_2_2.bias',
                           'conv_RA_1_1.weight', 'conv_RA_1_1.bias', 'conv_RA_1_2.weight', 'conv_RA_1_2.bias',
                           'conv_RA_2_1.weight', 'conv_RA_2_1.bias', 'conv_RA_2_2.weight', 'conv_RA_2_2.bias',
                           'conv_RL_1_1.weight', 'conv_RL_1_1.bias', 'conv_RL_1_2.weight',  'conv_RL_1_2.bias',
                           'conv_RL_2_1.weight', 'conv_RL_2_1.bias', 'conv_RL_2_2.weight', 'conv_RL_2_2.bias',
                           'fc3_LA.weight', 'fc3_LA.bias',
                           'fc3_N.weight', 'fc3_N.bias',
                           'fc3_RA.weight', 'fc3_RA.bias']
                           #'fc3_RL.weight', 'fc3_RL.bias',
                           #'fc3_LL.weight', 'fc3_LL.bias']

        for pn, pv in network.named_parameters():
            if pn in list_layers:
                pv.requires_grad = False

        return network



    ##################################################
    ###################  train  ######################
    ##################################################

    #def train(self, X_train_in, y_train_in, X_val_in, y_val_in, statistics, ea_itera, y_data_in=None):
    def train(self, ea_itera):
        '''
        Training and validating a network

        @param ea_itera: evolution iteration
        @return results_val: dict with validation results
        @return best_itera: best iteration when validating
        '''
        logging.info('        Network_User: Train---->')

        # Setting the network
        logging.info('        Network_User:    Train:    creating network')
        if self.config['network'] in ['cnn', 'cnn_imu', 'cnn_tpp', 'cnn_imu_tpp']:
            #network_obj = Network(self.config)
            self.network_obj.init_weights()

            # IF finetuning, load the weights from a source dataset
            if self.config["usage_modus"] == "fine_tuning":
                self.network_obj = self.load_weights(self.network_obj)

            # Displaying size of tensors
            logging.info('        Network_User:    Train:    network layers')
            for l in list(self.network_obj.named_parameters()):
                logging.info('        Network_User:    Train:    {} : {}'.format(l[0], l[1].cpu().detach().numpy().shape))

            logging.info('        Network_User:    Train:    setting device')
            self.network_obj.to(self.device)

        # Setting loss
        if self.config['output'] == 'softmax':
            logging.info('        Network_User:    Train:    setting criterion optimizer Softmax')
            if self.config["aggregate"] in ["FC", "FCN", "LSTM"]:
                criterion = nn.CrossEntropyLoss()
        elif self.config['output'] == 'attribute':
            logging.info('        Network_User:    Train:    setting criterion optimizer Attribute')
            if self.config["aggregate"] in ["FC", "FCN", "LSTM"]:
                criterion = nn.BCELoss()

        # Setting the freezing or not freezing from conv layers
        if self.config['freeze_options']:
            self.network_obj = self.set_required_grad(self.network_obj)

        # Setting optimizer
        optimizer = optim.RMSprop(self.network_obj.parameters(), lr=self.config['lr'], alpha=0.95)

        # Setting scheduler
        step_lr = self.config['epochs'] / 3
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=math.ceil(step_lr), gamma=0.5)

        if self.config['plotting']:
            # Plots

            logging.info('        Network_User:    Train:    setting plotting objects')

            fig = plt.figure()
            axis_list = []
            axis_list.append(fig.add_subplot(521, projection='3d'))
            axis_list.append(fig.add_subplot(522))
            axis_list.append(fig.add_subplot(523, projection='3d'))
            axis_list.append(fig.add_subplot(524))
            axis_list.append(fig.add_subplot(525, projection='3d'))
            axis_list.append(fig.add_subplot(526))
            axis_list.append(fig.add_subplot(527, projection='3d'))
            axis_list.append(fig.add_subplot(528))
            axis_list.append(fig.add_subplot(529, projection='3d'))

            plot_list = []
            # loss_plot, = axis_list[1].plot([], [],'-r',label='red')
            # plots acc, f1w, f1m for training
            plot_list.append(axis_list[1].plot([], [], '-r', label='acc')[0])
            plot_list.append(axis_list[1].plot([], [], '-b', label='f1w')[0])
            plot_list.append(axis_list[1].plot([], [], '-g', label='f1m')[0])

            # plot loss training
            plot_list.append(axis_list[3].plot([], [], '-r', label='loss tr')[0])

            # plots acc, f1w, f1m for training and validation
            plot_list.append(axis_list[5].plot([], [], '-r', label='acc tr')[0])
            plot_list.append(axis_list[5].plot([], [], '-b', label='f1w tr')[0])
            plot_list.append(axis_list[5].plot([], [], '-g', label='f1m tr')[0])

            plot_list.append(axis_list[5].plot([], [], '-c', label='acc vl')[0])
            plot_list.append(axis_list[5].plot([], [], '-m', label='f1w vl')[0])
            plot_list.append(axis_list[5].plot([], [], '-y', label='f1m vl')[0])

            # plot loss for training and validation
            plot_list.append(axis_list[7].plot([], [], '-r', label='loss tr')[0])
            plot_list.append(axis_list[7].plot([], [], '-b', label='loss vl')[0])

            # Customize the z axis.

            for al in range(len(axis_list)):
                if al % 2 == 0:
                    axis_list[al].set_zlim(0.0, 1.0)
                    axis_list[al].zaxis.set_major_locator(LinearLocator(10))
                    axis_list[al].zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Initializing lists for plots
        losses_train = []
        accs_train = []
        f1w_train = []
        f1m_train = []

        losses_val = []
        accs_val = []
        f1w_val = []
        f1m_val = []

        loss_train_val = []
        accs_train_val = []
        f1w_train_val = []
        f1m_train_val = []

        best_loss_val = np.inf

        #best_acc_val = 0

        # initialising object for computing metrics
        metrics_obj = Metrics(self.config, self.device, self.attrs)

        # loop for training
        itera = 0
        start_time_train = time.time()

        # zero the parameter gradients
        optimizer.zero_grad()

        best_itera = 0
        best_results_val = []
        itera = 0

        # loop for training
        # Validation is not carried out per epoch, but after certain number of iterations, specified
        # in configuration in main
        for e in range(self.config['epochs']):
            start_time_train = time.time()
            logging.info('\n        Network_User:    Train:    Training epoch {}'.format(e))
            start_time_batch = time.time()

            # loop per batch:
            for b, harwindow_batched in enumerate(self.dataLoader_train):
                start_time_batch = time.time()
                sys.stdout.write("\rTraining: Epoch {}/{} Batch {}/{} and itera {}".format(e,
                                                                                           self.config['epochs'],
                                                                                           b,
                                                                                           len(self.dataLoader_train),
                                                                                           itera))
                sys.stdout.flush()

                # Setting the network to train mode
                self.network_obj.train(mode=True)

                # Counting iterations
                # itera = (e * harwindow_batched["data"].shape[0]) + b

                # Selecting batch
                train_batch_v = harwindow_batched["data"]
                if self.config['output'] == 'softmax':
                    if self.config["aggregate"] in ["FCN", "LSTM"]:
                        train_batch_l = harwindow_batched["labels"][:, :]
                    elif self.config["aggregate"] == "FC":
                        #train_batch_l = harwindow_batched["label"][:, 0]
                        train_batch_l = harwindow_batched["label"]
                        train_batch_l = train_batch_l.reshape(-1)
                    elif self.config["aggregate"] == "LSTM2":
                        train_batch_l = harwindow_batched["labels"][:, :]
                elif self.config['output'] == 'attribute':
                    if self.config["aggregate"] in ["FCN", "LSTM"]:
                        train_batch_l = harwindow_batched["labels"][:, :]
                        train_batch_l_matrix = np.zeros((train_batch_l.shape[0], train_batch_l.shape[1],
                                                         self.config['num_attributes'] + 1))
                        for lx in range(train_batch_l.shape[0]):
                            for lattrx in range(train_batch_l.shape[1]):
                                train_batch_l_matrix[lx, lattrx, :] = self.attrs[train_batch_l[lx, lattrx]]
                    elif self.config["aggregate"] == "FC":
                        train_batch_l = harwindow_batched["label"]
                        train_batch_l_matrix = np.zeros((train_batch_l.shape[0], self.config['num_attributes'] + 1))
                        for lx in range(train_batch_l.shape[0]):
                            train_batch_l_matrix[lx, :] = self.attrs[train_batch_l[lx]]
                    elif self.config["aggregate"] == "LSTM2":
                        train_batch_l = harwindow_batched["labels"][:, :]



                # Creating torch tensors
                #train_batch_v = torch.from_numpy(train_batch_v)
                # if self.config['output'] == 'softmax':
                    #train_batch_l = torch.from_numpy(train_batch_l)
                #     train_batch_l = train_batch_l.type(dtype=torch.LongTensor)  #labels for crossentropy needs long type
                # elif self.config['output'] == 'attribute':
                #    train_batch_l = torch.from_numpy(train_batch_l_matrix)
                #    train_batch_l = train_batch_l.type(dtype=torch.FloatTensor)  #labels for crossentropy needs long type

                # Adding gaussian noise
                noise = self.normal.sample((train_batch_v.size()))
                noise = noise.reshape(train_batch_v.size())
                noise = noise.to(self.device, dtype=torch.float)

                # Sending to GPU
                train_batch_v = train_batch_v.to(self.device, dtype=torch.float)
                train_batch_v += noise
                if self.config['output'] == 'softmax':
                    train_batch_l = train_batch_l.to(self.device, dtype=torch.long) #labels for crossentropy needs long type
                elif self.config['output'] == 'attribute':
                    train_batch_l = torch.from_numpy(train_batch_l_matrix)
                    train_batch_l = train_batch_l.to(self.device, dtype=torch.float) #labels for binerycrossentropy needs float type

                # forward + backward + optimize
                feature_maps = self.network_obj(train_batch_v)
                if self.config["aggregate"] in ["FCN", "LSTM"]:
                    feature_maps = feature_maps.reshape(-1, feature_maps.size()[2])
                    if self.config['output'] == 'softmax':
                        train_batch_l = train_batch_l.reshape(-1)
                    if self.config['output'] == 'attribute':
                        train_batch_l = train_batch_l.reshape(-1, train_batch_l.size()[2])

                if self.config['output'] == 'softmax':
                    loss = criterion(feature_maps, train_batch_l) * (1 / self.config['accumulation_steps'])
                elif self.config['output'] == 'attribute':
                    if self.config["aggregate"] in ["FCN", "LSTM"]:
                        loss = criterion(feature_maps, train_batch_l[:, 1:]) * (1 / self.config['accumulation_steps'])
                    if self.config["aggregate"] == "FC":
                        loss = criterion(feature_maps, train_batch_l[:, 1:]) * (1 / self.config['accumulation_steps'])
                    if self.config["aggregate"] == "LSTM2":
                        loss = criterion(feature_maps, train_batch_l[:, 1:]) * (1 / self.config['accumulation_steps'])
                elif self.config['output'] == 'identity':
                    loss = criterion(feature_maps, train_batch_l) * (1 / self.config['accumulation_steps'])

                loss.backward()

                if (itera + 1) % self.config['accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(self.network_obj.parameters(), max_norm=10.0)
                    optimizer.step()
                    # zero the parameter gradients
                    optimizer.zero_grad()

                loss_train = loss.item()

                elapsed_time_batch = time.time() - start_time_batch

                #train_batch_l = harwindow_batched["label"]
                #train_batch_l = train_batch_l.reshape(-1)

                ################################## Validating ##################################################

                if (itera + 1) % self.config['valid_show'] == 0 or \
                        (itera) == (self.config['epochs'] * harwindow_batched["data"].shape[0]):
                    logging.info('\n')
                    logging.info('        Network_User:        Validating')
                    start_time_val = time.time()

                    # Setting the network to eval mode
                    self.network_obj.eval()

                    # Metrics for training for keeping the same number of metrics for val and training
                    # Metrics return a dict with the metrics.
                    #acc, f1_weighted, f1_mean, _ = metrics_obj.metric(targets=train_batch_l, predictions=feature_maps)
                    results_train = metrics_obj.metric(targets=train_batch_l, predictions=feature_maps)
                    loss_train_val.append(loss_train)
                    accs_train_val.append(results_train["classification"]['acc'])
                    f1w_train_val.append(results_train["classification"]['f1_weighted'])
                    f1m_train_val.append(results_train["classification"]['f1_mean'])

                    # Validation
                    results_val, loss_val = self.validate(self.network_obj, criterion)

                    elapsed_time_val = time.time() - start_time_val

                    # Appending the val metrics
                    losses_val.append(loss_val)
                    accs_val.append(results_val["classification"]['acc'])
                    f1w_val.append(results_val["classification"]['f1_weighted'])
                    f1m_val.append(results_val["classification"]['f1_mean'])
                    if self.config["sacred"]:
                        self.exp.log_scalar("Loss_TrainVal_inTrain_{}".format(ea_itera), value=loss_train, step=itera)
                        self.exp.log_scalar("Acc_TrainVal_inTrain_{}".format(ea_itera),
                                            value=results_train["classification"]['acc'], step=itera)
                        self.exp.log_scalar("F1w_TrainVal_inTrain_{}".format(ea_itera),
                                            value=results_train["classification"]['f1_weighted'], step=itera)
                        self.exp.log_scalar("F1m_TrainVal_inTrain_{}".format(ea_itera),
                                            value=results_train["classification"]['f1_mean'], step=itera)
                        self.exp.log_scalar("Loss_Val_inTrain_{}".format(ea_itera), value=loss_val, step=itera)
                        self.exp.log_scalar("Acc_Val_inTrain_{}".format(ea_itera).format(),
                                            value=results_val["classification"]['acc'],
                                            step=itera)
                        self.exp.log_scalar("F1w_Val_inTrain_{}".format(ea_itera),
                                            value=results_val["classification"]['f1_weighted'],
                                            step=itera)
                        self.exp.log_scalar("F1m_Val_inTrain_{}".format(ea_itera),
                                            value=results_val["classification"]['f1_mean'],
                                            step=itera)

                    # print statistics
                    logging.info('\n')
                    logging.info(
                        '        Network_User:        Validating:    '
                        'epoch {} batch {} itera {} elapsed time {}, best itera {}'.format(e, b, itera,
                                                                                           elapsed_time_val,
                                                                                           best_itera))
                    logging.info(
                        '        Network_User:        Validating:    '
                        'acc {}, f1_weighted {}, f1_mean {}'.format(results_val["classification"]['acc'],
                                                                    results_val["classification"]['f1_weighted'],
                                                                    results_val["classification"]['f1_mean']))

                    # Saving the network for the less iteration loss
                    if loss_val < best_loss_val: #results_val['f1_weighted'] > best_acc_val:
                        network_config = {
                            'NB_sensor_channels': self.config['NB_sensor_channels'],
                            'sliding_window_length': self.config['sliding_window_length'],
                            'filter_size': self.config['filter_size'],
                            'num_filters': self.config['num_filters'],
                            'reshape_input': self.config['reshape_input'],
                            'network': self.config['network'],
                            'output': self.config['output'],
                            'num_classes': self.config['num_classes'],
                            'num_attributes': self.config['num_attributes'],
                            'aggregate': self.config['aggregate'],
                            'labeltype': self.config['labeltype']
                        }

                        logging.info('        Network_User:            Saving the network')

                        torch.save({'state_dict': self.network_obj.state_dict(),
                                    'network_config': network_config,
                                    'att_rep': self.attr_representation},
                                   self.config['folder_exp'] + 'network.pt')
                        best_loss_val = loss_val
                        #best_acc_val = results_val["classification"]['f1_weighted']
                        best_results_val = results_val
                        best_itera = itera

                # Computing metrics for current training batch
                if (itera) % self.config['train_show'] == 0:
                    # Metrics for training
                    #acc, f1_weighted, f1_mean, _ = metrics_obj.metric(targets=train_batch_l, predictions=feature_maps)
                    results_train = metrics_obj.metric(targets=train_batch_l, predictions=feature_maps)

                    if self.config["sacred"]:
                        self.exp.log_scalar("Loss_Train_inTrain_{}".format(ea_itera), value=loss_train, step=itera)
                        self.exp.log_scalar("Acc_Train_inTrain_{}".format(ea_itera),
                                            value=results_train["classification"]['acc'], step=itera)
                        self.exp.log_scalar("F1w_Train_inTrain_{}".format(ea_itera),
                                            value=results_train["classification"]['f1_weighted'], step=itera)
                        self.exp.log_scalar("F1m_Train_inTrain_{}".format(ea_itera),
                                            value=results_train["classification"]['f1_mean'], step=itera)

                    activaciones = []
                    metrics_list = []
                    accs_train.append(results_train["classification"]['acc'])
                    f1w_train.append(results_train["classification"]['f1_weighted'])
                    f1m_train.append(results_train["classification"]['f1_mean'])
                    losses_train.append(loss_train)

                    if self.config['plotting']:
                        #For plotting
                        metrics_list.append(accs_train)
                        metrics_list.append(f1w_train)
                        metrics_list.append(f1m_train)
                        metrics_list.append(losses_train)
                        metrics_list.append(accs_train_val)
                        metrics_list.append(f1w_train_val)
                        metrics_list.append(f1m_train_val)
                        metrics_list.append(accs_val)
                        metrics_list.append(f1w_val)
                        metrics_list.append(f1m_val)
                        metrics_list.append(loss_train_val)
                        metrics_list.append(losses_val)
                        activaciones.append(train_batch_v.to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        #activaciones.append(feature_maps[0].to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        #activaciones.append(feature_maps[1].to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        #activaciones.append(feature_maps[2].to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        #activaciones.append(feature_maps[3].to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        self.plot(fig, axis_list, plot_list, metrics_list, activaciones,
                                  harwindow_batched["label"][:, 0][0].item(),
                                  torch.argmax(feature_maps[0], dim=0).item())

                    # print statistics
                    logging.info('        Network_User:            Dataset {} network {} pooling {} lr {} '
                                 'lr_optimizer {} Reshape {} '
                                 'Freeze {} Aggregator {}'.format(self.config["dataset"],
                                                                  self.config["network"],
                                                                  self.config["pooling"],
                                                                  self.config["lr"],
                                                                  optimizer.param_groups[0]['lr'],
                                                                  self.config["reshape_input"],
                                                                  self.config["freeze_options"],
                                                                  self.config["aggregate"]))
                    logging.info(
                        '        Network_User:    Train:    epoch {}/{} batch {}/{} itera {} '
                        'elapsed time {} best itera {}'.format(e, self.config['epochs'], b, len(self.dataLoader_train),
                                                               itera, elapsed_time_batch, best_itera))
                    logging.info('        Network_User:    Train:    loss {}'.format(loss))
                    logging.info(
                        '        Network_User:    Train:    acc {}, '
                        'f1_weighted {}, f1_mean {}'.format(results_train["classification"]['acc'],
                                                            results_train["classification"]['f1_weighted'],
                                                            results_train["classification"]['f1_mean']))
                    logging.info(
                        '        Network_User:    Train:    '
                        'Allocated {} GB Cached {} GB'.format(round(torch.cuda.memory_allocated(0)/1024**3, 1),
                                                              round(torch.cuda.memory_cached(0)/1024**3, 1)))
                    logging.info('\n\n--------------------------')
                itera += 1
            # Step of the the scheduler
            scheduler.step()

        # Validation
        results_val, loss_val = self.validate(self.network_obj, criterion)

        if loss_val < best_loss_val:
            logging.info('        Network_User:            Saving the network')

            torch.save({'state_dict': self.network_obj.state_dict(),
                        'network_config': network_config,
                        'att_rep': self.attrs},
                       self.config['folder_exp'] + 'network.pt')

        elapsed_time_train = time.time() - start_time_train

        logging.info('\n')
        logging.info(
            '        Network_User:    Train:    epoch {} batch {} itera {} '
            'Total training time {}'.format(e, b, itera, elapsed_time_train))

        # Storing the acc, f1s and losses of training and validation for the current training run
        np.savetxt(self.config['folder_exp'] + 'plots/acc_train_val.txt', accs_train_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1m_train_val.txt', f1m_train_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1w_train_val.txt', f1w_train_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/loss_train_val.txt', loss_train_val, delimiter=",", fmt='%s')

        np.savetxt(self.config['folder_exp'] + 'plots/acc_val.txt', accs_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1m_val.txt', f1m_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1w_val.txt', f1w_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/loss_val.txt', losses_val, delimiter=",", fmt='%s')

        np.savetxt(self.config['folder_exp'] + 'plots/acc_train.txt', accs_train, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1m_train.txt', f1m_train, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1w_train.txt', f1w_train, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/loss_train.txt', losses_train, delimiter=",", fmt='%s')

        del losses_train, accs_train, f1w_train, f1m_train
        del losses_val, accs_val, f1w_val, f1m_val
        del loss_train_val, accs_train_val, f1w_train_val, f1m_train_val
        del metrics_list, feature_maps

        torch.cuda.empty_cache()

        if self.config['plotting']:
            plt.savefig(self.config['folder_exp'] + 'training_final.png')
            plt.close()
                
        return best_results_val, best_itera





    ##################################################    
    ################  Validate  ######################
    ##################################################  

    def validate(self, network_obj, criterion):
        '''
        Validating a network

        @param network_obj: network object
        @param criterion: torch criterion object
        @return results_val: dict with validation results
        @return loss: loss of the validation
        '''

        # Setting the network to eval mode
        self.network_obj.eval()

        # Creating metric object
        metrics_obj = Metrics(self.config, self.device, self.attrs)
        loss_val = 0

        # One doesnt need the gradients
        with torch.no_grad():
            for v, harwindow_batched_val in enumerate(self.dataLoader_val):
                # Selecting batch
                test_batch_v = harwindow_batched_val["data"]
                #test_batch_l = harwindow_batched_val["label"]
                #test_batch_l = test_batch_l.reshape(-1)

                if self.config['output'] == 'softmax':
                    if self.config["aggregate"] in ["FCN", "LSTM"]:
                        test_batch_l = harwindow_batched_val["labels"][:, :]
                    elif self.config["aggregate"] == "FC":
                        test_batch_l = harwindow_batched_val["label"][:]
                        test_batch_l = test_batch_l.reshape(-1)
                    elif self.config["aggregate"] == "LSTM2":
                        test_batch_l = harwindow_batched_val["label"][:]
                        test_batch_l = test_batch_l.reshape(-1)
                if self.config['output'] == 'attribute':
                    if self.config["aggregate"] in ["FCN", "LSTM"]:
                        test_batch_l = harwindow_batched_val["labels"][:, :]
                        test_batch_l_matrix = np.zeros((test_batch_l.shape[0], test_batch_l.shape[1],
                                                         self.config['num_attributes'] + 1))
                        for lx in range(test_batch_l.shape[0]):
                            for lattrx in range(test_batch_l.shape[1]):
                                test_batch_l_matrix[lx, lattrx, :] = self.attrs[test_batch_l[lx, lattrx]]
                    elif self.config["aggregate"] == "FC":
                        test_batch_l = harwindow_batched_val["label"]
                        test_batch_l = test_batch_l.reshape(-1)
                        test_batch_l_matrix = np.zeros((test_batch_l.shape[0], self.config['num_attributes'] + 1))
                        for lx in range(test_batch_l.shape[0]):
                            test_batch_l_matrix[lx, :] = self.attrs[test_batch_l[lx]]

                # Creating torch tensors
                # test_batch_v = torch.from_numpy(test_batch_v)
                test_batch_v = test_batch_v.to(self.device, dtype=torch.float)
                if self.config['output'] == 'softmax':
                    test_batch_l = test_batch_l.to(self.device, dtype=torch.long) # labels for crossentropy needs long type
                elif self.config['output'] == 'attribute':
                    test_batch_l = torch.from_numpy(test_batch_l_matrix)
                    test_batch_l = test_batch_l.to(self.device, dtype=torch.float)

                # forward
                predictions = self.network_obj(test_batch_v)

                if self.config["aggregate"] in ["FCN", "LSTM"]:
                    #predictions = predictions.reshape(-1, predictions.size()[2])
                    if self.config['output'] == 'softmax':
                        test_batch_l = test_batch_l.reshape(-1)
                    elif self.config['output'] == 'attribute':
                        test_batch_l = test_batch_l.reshape(-1, test_batch_l.size()[2])

                if self.config['output'] == 'softmax':
                    loss = criterion(predictions, test_batch_l)
                elif self.config['output'] == 'attribute':
                    if self.config["aggregate"] in ["FCN", "LSTM"]:
                        loss = criterion(predictions, test_batch_l[:, 1:])
                    elif self.config["aggregate"] == "FC":
                        loss = criterion(predictions, test_batch_l[:, 1:])
                    elif self.config["aggregate"] == "LSTM2":
                        loss = criterion(predictions, test_batch_l[:, 1:])
                elif self.config['output'] == 'identity':
                    loss = criterion(predictions, test_batch_l)

                loss_val = loss_val + loss.item()

                # As creating an empty tensor and sending to device and then concatenating isnt working
                if v == 0:
                    predictions_val = predictions
                    test_labels = test_batch_l
                else:
                    predictions_val = torch.cat((predictions_val, predictions), dim=0)
                    test_labels = torch.cat((test_labels, test_batch_l), dim=0)

                sys.stdout.write("\rValidating: Batch  %i" % v)
                sys.stdout.flush()

        print("\n")
        # Computing metrics of validation
        results_val = metrics_obj.metric(test_labels, predictions_val)

        del test_batch_v, test_batch_l
        del predictions, predictions_val

        torch.cuda.empty_cache()

        return results_val, loss_val / v



    ##################################################    
    ###################  test  ######################
    ##################################################

    def test(self, ea_itera, ifVal=False):
        '''
        Testing a network

        @param ea_itera: evolution iteration
        @return results_test: dict with testing results
        @return confusion matrix: confusion matrix of testing results
        '''

        logging.info('        Network_User:    Test ---->')

        logging.info('        Network_User:    Test:    creating network')
        if self.config['network'] in ['cnn', 'cnn_imu', 'cnn_tpp', 'cnn_imu_tpp']:
            network_obj = Network(self.config)

            # Loading the model
            network_obj.load_state_dict(torch.load(self.config['folder_exp'] + 'network.pt')['state_dict'])
            network_obj.eval()

            logging.info('        Network_User:    Test:    setting device')
            network_obj.to(self.device)

            # Setting loss
            if self.config['output'] == 'softmax':
                logging.info('        Network_User:    Train:    setting criterion optimizer Softmax')
                criterion = nn.CrossEntropyLoss()
            elif self.config['output'] == 'attribute':
                logging.info('        Network_User:    Train:    setting criterion optimizer Attribute')
                criterion = nn.BCELoss()

        loss_test = 0

        logging.info('        Network_User:    Testing')
        start_time_test = time.time()

        metrics_obj = Metrics(self.config, self.device, self.attrs)

        # loop for testing
        with torch.no_grad():
            for v, harwindow_batched_test in enumerate(self.dataLoader_test):
                # Selecting batch
                test_batch_v = harwindow_batched_test["data"]
                if self.config['output'] == 'softmax':
                    if self.config["aggregate"] in ["FCN", "LSTM"]:
                        test_batch_l = harwindow_batched_test["labels"][:, :]

                        test_labels_window_batch = harwindow_batched_test["labels"]
                    elif self.config["aggregate"] == "FC":
                        test_batch_l = harwindow_batched_test["label"]
                        test_batch_l = test_batch_l.reshape(-1)

                        test_labels_window_batch = harwindow_batched_test["labels"]
                    elif self.config["aggregate"] == "LSTM2":
                        test_batch_l = harwindow_batched_test["label"]
                        test_batch_l = test_batch_l.reshape(-1)

                        test_labels_window_batch = harwindow_batched_test["labels"]
                elif self.config['output'] == 'attribute':
                    if self.config["aggregate"] in ["FCN", "LSTM"]:
                        test_batch_l = harwindow_batched_test["labels"][:, :]

                        test_labels_window_batch = harwindow_batched_test["labels"]
                        test_batch_l_matrix = np.zeros((test_batch_l.shape[0], test_batch_l.shape[1],
                                                         self.config['num_attributes'] + 1))
                        for lx in range(test_batch_l.shape[0]):
                            for lattrx in range(test_batch_l.shape[1]):
                                test_batch_l_matrix[lx, lattrx, :] = self.attrs[test_batch_l[lx, lattrx]]
                    elif self.config["aggregate"] == "FC":
                        test_batch_l = harwindow_batched_test["label"]

                        test_labels_window_batch = harwindow_batched_test["labels"]
                        test_batch_l_matrix = np.zeros((test_batch_l.shape[0], self.config['num_attributes'] + 1))
                        for lx in range(test_batch_l.shape[0]):
                            test_batch_l_matrix[lx, :] = self.attrs[test_batch_l[lx]]
                    elif self.config["aggregate"] == "LSTM2":
                        test_batch_l = harwindow_batched_test["label"]

                        test_labels_window_batch = harwindow_batched_test["labels"]


                # Creating torch tensors
                test_batch_v = test_batch_v.to(self.device, dtype=torch.float)
                if self.config['output'] == 'softmax':
                    test_batch_l = test_batch_l.to(self.device, dtype=torch.long) # labels for crossentropy needs long type
                elif self.config['output'] == 'attribute':
                    test_batch_l = torch.from_numpy(test_batch_l_matrix)
                    test_batch_l = test_batch_l.to(self.device, dtype=torch.float)

                #forward
                predictions = network_obj(test_batch_v)

                if self.config["aggregate"] in ["FCN", "LSTM"]:
                    #predictions = predictions.reshape(-1, predictions.size()[2])
                    if self.config['output'] == 'softmax':
                        test_batch_l = test_batch_l.reshape(-1)
                    elif self.config['output'] == 'attribute':
                        test_batch_l = test_batch_l.reshape(-1, test_batch_l.size()[2])

                if self.config['output'] == 'softmax':
                    loss = criterion(predictions, test_batch_l)
                elif self.config['output'] == 'attribute':
                    if self.config["aggregate"] in ["FCN", "LSTM"]:
                        loss = criterion(predictions, test_batch_l[:, 1:])
                    elif self.config["aggregate"] == "FC":
                        loss = criterion(predictions, test_batch_l[:, 1:])
                    elif self.config["aggregate"] == "LSTM2":
                        loss = criterion(predictions, test_batch_l[:, 1:])
                loss_test = loss_test + loss.item()

                # As creating an empty tensor and sending to device and then concatenating isnt working
                if v == 0:
                    predictions_test = predictions
                    test_labels = test_batch_l
                    test_labels_window = test_labels_window_batch
                else:
                    predictions_test = torch.cat((predictions_test, predictions), dim=0)
                    test_labels = torch.cat((test_labels, test_batch_l), dim=0)
                    test_labels_window = torch.cat((test_labels_window, test_labels_window_batch), dim=0)

                sys.stdout.write("\rTesting: Batch  {}/{}".format(v, len(self.dataLoader_test)))
                sys.stdout.flush()

        elapsed_time_test = time.time() - start_time_test

        #Computing metrics
        logging.info('            Train:    type targets vector: {}'.format(test_labels.type()))
        results_test = metrics_obj.metric(test_labels, predictions_test, mode="classification")

        #print statistics
        logging.info(
            '        Network_User:        Testing:    elapsed time {} acc {}, f1_weighted {}, f1_mean {}'.format(
                elapsed_time_test, results_test["classification"]['acc'], results_test["classification"]['f1_weighted'],
                results_test["classification"]['f1_mean']))

        predictions_labels = results_test["classification"]['predicted_classes'].to("cpu", torch.double).numpy()
        test_labels = test_labels.to("cpu", torch.double).numpy()
        if self.config['output'] == 'softmax':
            test_labels = test_labels
        elif self.config['output'] == 'attribute':
            test_labels = test_labels[:, 0]
        elif self.config['output'] == 'identity':
            test_labels = test_labels

        #Segmenting
        if self.config["aggregate"] in ["FCN", "LSTM"]:
            predictions_segment = torch.reshape(results_test["classification"]['predicted_classes'],
                                                shape=(test_labels_window.size()))
            sample_targets_single, sample_predictions_single = self.setting_samples_from_windows(test_labels_window,
                                                                                                 predictions_segment)

        else:
            sample_targets_single, sample_predictions_single = self.setting_samples_from_windows(test_labels_window,
                                                               results_test["classification"]['predicted_classes'])

        #Computing metrics
        sample_targets_single = sample_targets_single.to(self.device, dtype=torch.long)
        sample_predictions_single = sample_predictions_single.to(self.device, dtype=torch.long)
        logging.info('            Train:    type targets vector: {}'.format(sample_targets_single.type()))
        results_test_segment = metrics_obj.metric(sample_targets_single, sample_predictions_single, mode="segmentation")
        sample_targets_single = sample_targets_single.to("cpu", torch.double).numpy()
        sample_predictions_single = sample_predictions_single.to("cpu", torch.double).numpy()

        #print statistics
        logging.info(
            '        Network_User:        Testing Segmentation:    elapsed time {} acc {}, '
            'f1_weighted {}, f1_mean {}'.format(elapsed_time_test, results_test_segment["segmentation"]['acc'],
                                                results_test_segment["segmentation"]['f1_weighted'],
                                                results_test_segment["segmentation"]['f1_mean']))

        # Computing confusion matrix
        confusion_matrix = np.zeros((self.config['num_classes'], self.config['num_classes']))

        for cl in range(self.config['num_classes']):
            pos_pred = predictions_labels == cl
            pos_pred_trg = np.reshape(test_labels, newshape=test_labels.shape[0])[pos_pred]
            bincount = np.bincount(pos_pred_trg.astype(int), minlength=self.config['num_classes'])
            confusion_matrix[cl, :] = bincount

        logging.info("        Network_User:        Testing:    Confusion matrix \n{}\n".format(confusion_matrix.astype(int)))

        percentage_pred = []
        for cl in range(self.config['num_classes']):
            pos_trg = np.reshape(test_labels, newshape=test_labels.shape[0]) == cl
            percentage_pred.append(confusion_matrix[cl, cl] / float(np.sum(pos_trg)))
        percentage_pred = np.array(percentage_pred)

        logging.info("        Network_User:        Validating:    percentage Pred \n{}\n".format(percentage_pred))

        del test_batch_v, test_batch_l
        del predictions, predictions_test
        del test_labels, predictions_labels
        del network_obj
        
        torch.cuda.empty_cache()
        
        return results_test_segment, confusion_matrix.astype(int)

    ##################################################    
    ############  evolution_evaluation  ##############
    ##################################################
    
    def evolution_evaluation(self, ea_iter, testing=False):
        '''
        Organises the evolution, training, testing or validating

        @param ea_itera: evolution iteration
        @param testing: Setting testing in training or only testing
        @return results: dict with validating/testing results
        @return confusion_matrix: dict with validating/testing results
        @return best_itera: best iteration for training
        '''

        logging.info('        Network_User: Evolution evaluation iter {}'.format(ea_iter))

        #acc_test = 0
        #f1_weighted_test = 0
        #f1_mean_test = 0
        confusion_matrix = 0
        best_itera = 0
        if testing:
            logging.info('        Network_User: Testing')
            results, confusion_matrix = self.test(ea_iter)
        else:
            if self.config['usage_modus'] == 'train':
                logging.info('        Network_User: Training')

                results_train, best_itera = self.train(ea_iter)
                results, _ = self.test(ea_iter, ifVal=True)

            elif self.config['usage_modus'] == 'evolution':
                logging.info('        Network_User: Evolution')

            elif self.config['usage_modus'] == 'train_final':
                logging.info('        Network_User: Final Training')
                results, best_itera = self.train(ea_iter)
                results, _ = self.test(ea_iter, ifVal=True)

            elif self.config['usage_modus'] == 'fine_tuning':
                logging.info('        Network_User: Fine Tuning')
                results, best_itera = self.train(ea_iter)
                results, _ = self.test(ea_iter, ifVal=True)

            elif self.config['usage_modus'] == 'test':
                logging.info('        Network_User: Testing')

                results, confusion_matrix = self.test(ea_iter)

            else:
                logging.info('        Network_User: Not selected modus')

        return results, confusion_matrix, best_itera
