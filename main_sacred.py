'''
Created on Jun 08, 2020

@author: fmoya
'''

from __future__ import print_function
import os
import logging
import torch
import numpy as np
import random

import platform
from modus_selecter import Modus_Selecter

import datetime

from sacred import Experiment
#from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

def load_credentials(path='~/.mongodb_credentials'):
    path = os.path.expanduser(path)
 
    logger = logging.getLogger('::load_credentials')
    logger.info(f'Loading credientials from {path}')
    with io.open(path) as f:
        user, pw, url, db_name = f.read().strip().split(',')
 
    return user, pw, url, db_name

user, pw, url, db_name = load_credentials(path='~/.mongodb_credentials')


ex = Experiment()
# ex.captured_out_filter = apply_backspaces_and_linefeeds

ex.observers.append(MongoObserver.create(url='curtis',
                                         db_name='your database',
                                         username='your user account for mongodb',
                                         password='your passwod for mongodb',
                                         authSource='admin',
                                         authMechanism='SCRAM-SHA-1'))


def configuration(dataset_idx, network_idx, output_idx, usage_modus_idx=0, dataset_fine_tuning_idx=0,
                  reshape_input=False, learning_rates_idx=0, name_counter=0, freeze=0, percentage_idx=0,
                  fully_convolutional=False, pooling=0, sacred=True):
    """
    Set a configuration of all the possible variables that were set in the experiments.
    This includes the datasets, hyperparameters for training, networks, outputs, datasets paths,
    results paths

    @param dataset_idx: id of dataset
    @param network_idx: id of network 0 for tcnn, 1, for tcnn-lstm, 2 tcnn-IMU
    @param output_idx: 0 for softmax, 1 for attributes
    @param usage_modus_idx: id of the modus 0 for train, 1 for test, 2 for evolution, 3 for train_final,...
    @param dataset_fine_tuning_idx: id of source dataset in case of finetuning
    @param reshape_input: reshaping the input False for [C,T] or, True for [3,C/3,T]=[[x,y,z], [sensors], Time]
    @param learning_rates_idx: id for the Learning Rate
    @param name_counter: counter for experiments
    @param name_counter: 0 for freezing the CNN layers, or 1 for fine-tuning
    @param percentage_idx: Percentage for the training dataset
    @param fully_convolutional: False for FC or True for FCN
    @return: configuration: dict with all the configurations
    """

    # Flags
    plotting = False

    # Options
    dataset = {0: 'mocap', 1: 'mbientlab', 2: 'virtual', 3: 'mocap_half', 4: 'virtual_quarter', 5: 'mocap_quarter',
               6: 'mbientlab_50_p', 7: 'mbientlab_10_p', 8: 'mbientlab_50_r', 9: 'mbientlab_10_r',
               10: 'mbientlab_quarter', 11: 'motionminers_real', 12: 'motionminers_flw'}
    network = {0: 'cnn', 1: 'lstm', 2: 'cnn_imu', 3: 'cnn_tpp', 4: 'cnn_imu_tpp'}
    output = {0: 'softmax', 1: 'attribute', 2: 'identity'}
    usage_modus = {0: 'train', 1: 'test', 2: 'evolution', 3: 'train_final', 4: 'train_random', 5: 'fine_tuning'}

    # Dataset Hyperparameters
    NB_sensor_channels = {'mocap': 126, 'mbientlab': 30, 'virtual': 126, 'mocap_half': 126, 'virtual_quarter': 126,
                          'mocap_quarter': 126, 'mbientlab_50_p': 30, 'mbientlab_10_p': 30, 'mbientlab_50_r': 30,
                          'mbientlab_10_r': 30, 'mbientlab_quarter': 30, 'motionminers_real': 27,
                          'motionminers_flw': 27}
    sliding_window_length = {'mocap': 200, 'mbientlab': 100, 'virtual': 100, 'mocap_half': 100, 'virtual_quarter': 25,
                             'mocap_quarter': 25, 'mbientlab_50_p': 100, 'mbientlab_10_p': 100, 'mbientlab_50_r': 100,
                             'mbientlab_10_r': 100, 'mbientlab_quarter': 25, 'motionminers_real': 100,
                             'motionminers_flw': 100}
    sliding_window_step = {'mocap': 25, 'mbientlab': 12, 'virtual': 12, 'mocap_half': 12, 'virtual_quarter': 12,
                           'mocap_quarter': 12, 'mbientlab_50_p': 12, 'mbientlab_10_p': 12, 'mbientlab_50_r': 12,
                           'mbientlab_10_r': 12, 'mbientlab_quarter': 12, 'motionminers_real': 12,
                           'motionminers_flw': 12}
    num_attributes = {'mocap': 19, 'mbientlab': 19, 'virtual': 19, 'mocap_half': 19, 'virtual_quarter': 19,
                      'mocap_quarter': 19, 'mbientlab_50_p': 19, 'mbientlab_10_p': 19, 'mbientlab_50_r': 19,
                      'mbientlab_10_r': 19, 'mbientlab_quarter': 19, 'motionminers_real': 19,
                      'motionminers_flw': 19}
    num_tr_inputs = {'mocap': 247702, 'mbientlab': 91399, 'virtual': 239013, 'mocap_half': 213472,
                     'virtual_quarter': 116428, 'mocap_quarter': 168505, 'mbientlab_50_p': 49850,
                     'mbientlab_10_p': 27591, 'mbientlab_50_r': 21791, 'mbientlab_10_r': 8918,
                     'mbientlab_quarter': 91384, 'motionminers_real': 22282, 'motionminers_flw': 93712}

    if output[output_idx] == 'identity':
        num_classes = {'mocap': 14, 'mbientlab': 14, 'virtual': 14, 'mocap_half': 14, 'virtual_quarter': 14,
                       'mocap_quarter': 14, 'mbientlab_50_p': 14, 'mbientlab_10_p': 14, 'mbientlab_50_r': 14,
                       'mbientlab_10_r': 14, 'mbientlab_quarter': 14, 'motionminers_real': 3, 'motionminers_flw': 14}
    else:
        num_classes = {'mocap': 7, 'mbientlab': 7, 'virtual': 7, 'mocap_half': 7, 'virtual_quarter': 7,
                       'mocap_quarter': 7, 'mbientlab_50_p': 7, 'mbientlab_10_p': 7, 'mbientlab_50_r': 7,
                       'mbientlab_10_r': 7, 'mbientlab_quarter': 7, 'motionminers_real': 6, 'motionminers_flw': 7}

    # Learning rate
    learning_rates = [0.0001, 0.00001, 0.000001]
    lr = {'mocap': {'cnn': learning_rates[learning_rates_idx],
                    'lstm': learning_rates[learning_rates_idx],
                    'cnn_imu': learning_rates[learning_rates_idx],
                    'cnn_tpp': learning_rates[learning_rates_idx],
                    'cnn_imu_tpp': learning_rates[learning_rates_idx]},
          'mbientlab': {'cnn': learning_rates[learning_rates_idx],
                        'lstm': learning_rates[learning_rates_idx],
                        'cnn_imu': learning_rates[learning_rates_idx],
                        'cnn_tpp': learning_rates[learning_rates_idx],
                        'cnn_imu_tpp': learning_rates[learning_rates_idx]},
          'virtual': {'cnn': learning_rates[learning_rates_idx],
                      'lstm': learning_rates[learning_rates_idx],
                      'cnn_imu': learning_rates[learning_rates_idx],
                      'cnn_tpp': learning_rates[learning_rates_idx],
                      'cnn_imu_tpp': learning_rates[learning_rates_idx]},
          'mocap_half': {'cnn': learning_rates[learning_rates_idx],
                         'lstm': learning_rates[learning_rates_idx],
                         'cnn_imu': learning_rates[learning_rates_idx],
                         'cnn_tpp': learning_rates[learning_rates_idx],
                         'cnn_imu_tpp': learning_rates[learning_rates_idx]},
          'virtual_quarter': {'cnn': learning_rates[learning_rates_idx],
                              'lstm': learning_rates[learning_rates_idx],
                              'cnn_imu': learning_rates[learning_rates_idx],
                              'cnn_tpp': learning_rates[learning_rates_idx],
                              'cnn_imu_tpp': learning_rates[learning_rates_idx]},
          'mocap_quarter': {'cnn': learning_rates[learning_rates_idx],
                            'lstm': learning_rates[learning_rates_idx],
                            'cnn_imu': learning_rates[learning_rates_idx],
                            'cnn_tpp': learning_rates[learning_rates_idx],
                            'cnn_imu_tpp': learning_rates[learning_rates_idx]},
          'mbientlab_50_p': {'cnn': learning_rates[learning_rates_idx],
                             'lstm': learning_rates[learning_rates_idx],
                             'cnn_imu': learning_rates[learning_rates_idx],
                             'cnn_tpp': learning_rates[learning_rates_idx],
                             'cnn_imu_tpp': learning_rates[learning_rates_idx]},
          'mbientlab_10_p': {'cnn': learning_rates[learning_rates_idx],
                             'lstm': learning_rates[learning_rates_idx],
                             'cnn_imu': learning_rates[learning_rates_idx],
                             'cnn_tpp': learning_rates[learning_rates_idx],
                             'cnn_imu_tpp': learning_rates[learning_rates_idx]},
          'mbientlab_50_r': {'cnn': learning_rates[learning_rates_idx],
                             'lstm': learning_rates[learning_rates_idx],
                             'cnn_imu': learning_rates[learning_rates_idx],
                             'cnn_tpp': learning_rates[learning_rates_idx],
                             'cnn_imu_tpp': learning_rates[learning_rates_idx]},
          'mbientlab_10_r': {'cnn': learning_rates[learning_rates_idx],
                             'lstm': learning_rates[learning_rates_idx],
                             'cnn_imu': learning_rates[learning_rates_idx],
                             'cnn_tpp': learning_rates[learning_rates_idx],
                             'cnn_imu_tpp': learning_rates[learning_rates_idx]},
          'mbientlab_quarter': {'cnn': learning_rates[learning_rates_idx],
                                'lstm': learning_rates[learning_rates_idx],
                                'cnn_imu': learning_rates[learning_rates_idx],
                                'cnn_tpp': learning_rates[learning_rates_idx],
                                'cnn_imu_tpp': learning_rates[learning_rates_idx]},
          'motionminers_real': {'cnn': learning_rates[learning_rates_idx],
                                'lstm': learning_rates[learning_rates_idx],
                                'cnn_imu': learning_rates[learning_rates_idx],
                                'cnn_tpp': learning_rates[learning_rates_idx],
                                'cnn_imu_tpp': learning_rates[learning_rates_idx]},
          'motionminers_flw': {'cnn': learning_rates[learning_rates_idx],
                               'lstm': learning_rates[learning_rates_idx],
                               'cnn_imu': learning_rates[learning_rates_idx],
                               'cnn_tpp': learning_rates[learning_rates_idx],
                               'cnn_imu_tpp': learning_rates[learning_rates_idx]}
          }
    lr_mult = 1.0

    # Maxout
    use_maxout = {'cnn': False, 'lstm': False, 'cnn_imu': False, 'cnn_tpp': False, 'cnn_imu_tpp': False}

    # Balacing
    balancing = {'mocap': False, 'mbientlab': False, 'virtual': False, 'mocap_half': False, 'virtual_quarter': False,
                 'mocap_quarter': False, 'mbientlab_50_p': False, 'mbientlab_10_p': False, 'mbientlab_50_r': False,
                 'mbientlab_10_r': False, 'mbientlab_quarter': False, 'motionminers_real': False,
                 'motionminers_flw': False}

    # Epochs
    if usage_modus[usage_modus_idx] == 'train_final' or usage_modus[usage_modus_idx] == 'fine_tuning':
        epoch_mult = 2
    else:
        epoch_mult = 1

    epochs = {'mocap': {'cnn': {'softmax': 6, 'attribute': 6, 'identity': 5},
                        'lstm': {'softmax': 6, 'attribute': 6, 'identity': 5},
                        'cnn_imu': {'softmax': 6, 'attribute': 6, 'identity': 5},
                        'cnn_tpp': {'softmax': 6, 'attribute': 6, 'identity': 5},
                        'cnn_imu_tpp': {'softmax': 6, 'attribute': 6, 'identity': 5}},
              'mbientlab': {'cnn': {'softmax': 10, 'attribute': 10, 'identity': 10},
                            'lstm': {'softmax': 10, 'attribute': 10, 'identity': 10},
                            'cnn_imu': {'softmax': 10, 'attribute': 10, 'identity': 10},
                            'cnn_tpp': {'softmax': 10, 'attribute': 10, 'identity': 10},
                            'cnn_imu_tpp': {'softmax': 10, 'attribute': 10, 'identity': 10}},
              'virtual': {'cnn': {'softmax': 32, 'attribute': 50, 'identity': 10},
                          'lstm': {'softmax': 10, 'attribute': 5, 'identity': 10},
                          'cnn_imu': {'softmax': 32, 'attribute': 50, 'identity': 10},
                          'cnn_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10},
                          'cnn_imu_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10}},
              'mocap_half': {'cnn': {'softmax': 32, 'attribute': 50, 'identity': 10},
                             'lstm': {'softmax': 10, 'attribute': 5, 'identity': 10},
                             'cnn_imu': {'softmax': 32, 'attribute': 50, 'identity': 10},
                             'cnn_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10},
                             'cnn_imu_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10}},
              'virtual_quarter': {'cnn': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                  'lstm': {'softmax': 10, 'attribute': 5, 'identity': 10},
                                  'cnn_imu': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                  'cnn_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                  'cnn_imu_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10}},
              'mocap_quarter': {'cnn': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                'lstm': {'softmax': 10, 'attribute': 5, 'identity': 10},
                                'cnn_imu': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                'cnn_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                'cnn_imu_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10}},
              'mbientlab_50_p': {'cnn': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                 'lstm': {'softmax': 10, 'attribute': 5, 'identity': 10},
                                 'cnn_imu': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                 'cnn_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                 'cnn_imu_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10}},
              'mbientlab_10_p': {'cnn': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                 'lstm': {'softmax': 10, 'attribute': 5, 'identity': 10},
                                 'cnn_imu': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                 'cnn_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                 'cnn_imu_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10}},
              'mbientlab_50_r': {'cnn': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                 'lstm': {'softmax': 10, 'attribute': 5, 'identity': 10},
                                 'cnn_imu': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                 'cnn_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                 'cnn_imu_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10}},
              'mbientlab_10_r': {'cnn': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                 'lstm': {'softmax': 10, 'attribute': 5, 'identity': 10},
                                 'cnn_imu': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                 'cnn_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                 'cnn_imu_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10}},
              'mbientlab_quarter': {'cnn': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                    'lstm': {'softmax': 10, 'attribute': 5, 'identity': 10},
                                    'cnn_imu': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                    'cnn_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10},
                                    'cnn_imu_tpp': {'softmax': 32, 'attribute': 50, 'identity': 10}},
              'motionminers_real': {'cnn': {'softmax': 20, 'attribute': 10, 'identity': 10},
                                    'lstm': {'softmax': 10, 'attribute': 10, 'identity': 10},
                                    'cnn_imu': {'softmax': 10, 'attribute': 10, 'identity': 10},
                                    'cnn_tpp': {'softmax': 10, 'attribute': 10, 'identity': 10},
                                    'cnn_imu_tpp': {'softmax': 10, 'attribute': 10, 'identity': 10}},
              'motionminers_flw': {'cnn': {'softmax': 10, 'attribute': 10, 'identity': 10},
                                   'lstm': {'softmax': 10, 'attribute': 10, 'identity': 10},
                                   'cnn_imu': {'softmax': 10, 'attribute': 10, 'identity': 10},
                                   'cnn_tpp': {'softmax': 10, 'attribute': 10, 'identity': 10},
                                   'cnn_imu_tpp': {'softmax': 10, 'attribute': 10, 'identity': 10}}
              }

    division_epochs = {'mocap': 2, 'mbientlab': 1, 'virtual': 1, 'mocap_half': 1, 'virtual_quarter': 1,
                       'mocap_quarter': 1, 'mbientlab_50_p': 1, 'mbientlab_10_p': 1, 'mbientlab_50_r': 1,
                       'mbientlab_10_r': 1, 'mbientlab_quarter': 1, 'motionminers_real': 1,
                       'motionminers_flw': 1}

    # Batch size
    batch_size_train = {
        'cnn': {'mocap': 100, 'mbientlab': 100, 'virtual': 100, 'mocap_half': 100, 'virtual_quarter': 100,
                'mocap_quarter': 100, 'mbientlab_50_p': 100, 'mbientlab_10_p': 100, 'mbientlab_50_r': 100,
                'mbientlab_10_r': 25, 'mbientlab_quarter': 100, 'motionminers_real': 100, 'motionminers_flw': 100},
        'lstm': {'mocap': 100, 'mbientlab': 100, 'virtual': 100, 'mocap_half': 100, 'virtual_quarter': 100,
                 'mocap_quarter': 100, 'mbientlab_50_p': 100, 'mbientlab_10_p': 100, 'mbientlab_50_r': 100,
                 'mbientlab_10_r': 100, 'mbientlab_quarter': 100, 'motionminers_real': 100, 'motionminers_flw': 100},
        'cnn_imu': {'mocap': 100, 'mbientlab': 100, 'virtual': 100, 'mocap_half': 100, 'virtual_quarter': 100,
                    'mocap_quarter': 100, 'mbientlab_50_p': 100, 'mbientlab_10_p': 100, 'mbientlab_50_r': 100,
                    'mbientlab_10_r': 25, 'mbientlab_quarter': 100, 'motionminers_real': 100, 'motionminers_flw': 100},
        'cnn_tpp': {'mocap': 100, 'mbientlab': 100, 'virtual': 100, 'mocap_half': 100, 'virtual_quarter': 100,
                    'mocap_quarter': 100, 'mbientlab_50_p': 100, 'mbientlab_10_p': 100, 'mbientlab_50_r': 100,
                    'mbientlab_10_r': 25, 'mbientlab_quarter': 100, 'motionminers_real': 100, 'motionminers_flw': 100},
        'cnn_imu_tpp': {'mocap': 100, 'mbientlab': 100, 'virtual': 100, 'mocap_half': 100, 'virtual_quarter': 100,
                        'mocap_quarter': 100, 'mbientlab_50_p': 100, 'mbientlab_10_p': 100, 'mbientlab_50_r': 100,
                        'mbientlab_10_r': 25, 'mbientlab_quarter': 100, 'motionminers_real': 100,
                        'motionminers_flw': 100}
    }

    batch_size_val = {'cnn': {'mocap': 100, 'mbientlab': 200, 'virtual': 100, 'mocap_half': 100,
                              'virtual_quarter': 100, 'mocap_quarter': 100, 'mbientlab_50_p': 100,
                              'mbientlab_10_p': 100, 'mbientlab_50_r': 100, 'mbientlab_10_r': 25,
                              'mbientlab_quarter': 100, 'motionminers_real': 200, 'motionminers_flw': 200},
                      'lstm': {'mocap': 100, 'mbientlab': 200, 'virtual': 100, 'mocap_half': 100,
                               'virtual_quarter': 100, 'mocap_quarter': 100, 'mbientlab_50_p': 100,
                               'mbientlab_10_p': 100, 'mbientlab_50_r': 100, 'mbientlab_10_r': 100,
                               'mbientlab_quarter': 100, 'motionminers_real': 200, 'motionminers_flw': 200},
                      'cnn_imu': {'mocap': 100, 'mbientlab': 200, 'virtual': 100, 'mocap_half': 100,
                                  'virtual_quarter': 100, 'mocap_quarter': 100, 'mbientlab_50_p': 100,
                                  'mbientlab_10_p': 100, 'mbientlab_50_r': 100, 'mbientlab_10_r': 25,
                                  'mbientlab_quarter': 100, 'motionminers_real': 200, 'motionminers_flw': 200},
                      'cnn_tpp': {'mocap': 100, 'mbientlab': 200, 'virtual': 100, 'mocap_half': 100,
                                  'virtual_quarter': 100, 'mocap_quarter': 100, 'mbientlab_50_p': 100,
                                  'mbientlab_10_p': 100, 'mbientlab_50_r': 100, 'mbientlab_10_r': 25,
                                  'mbientlab_quarter': 100, 'motionminers_real': 200, 'motionminers_flw': 200},
                      'cnn_imu_tpp': {'mocap': 100, 'mbientlab': 200, 'virtual': 100, 'mocap_half': 100,
                                      'virtual_quarter': 100, 'mocap_quarter': 100, 'mbientlab_50_p': 100,
                                      'mbientlab_10_p': 100, 'mbientlab_50_r': 100, 'mbientlab_10_r': 25,
                                      'mbientlab_quarter': 100, 'motionminers_real': 200, 'motionminers_flw': 200}
                      }

    accumulation_steps = {'mocap': 4, 'mbientlab': 4, 'virtual': 4, 'mocap_half': 4, 'virtual_quarter': 4,
                          'mocap_quarter': 4, 'mbientlab_50_p': 4, 'mbientlab_10_p': 4, 'mbientlab_50_r': 4,
                          'mbientlab_10_r': 4, 'mbientlab_quarter': 4, 'motionminers_real': 4, 'motionminers_flw': 4}

    # Filters
    filter_size = {'mocap': 5, 'mbientlab': 5, 'virtual': 5, 'mocap_half': 5, 'virtual_quarter': 5, 'mocap_quarter': 5,
                   'mbientlab_50_p': 5, 'mbientlab_10_p': 5, 'mbientlab_50_r': 5, 'mbientlab_10_r': 5,
                   'mbientlab_quarter': 5, 'motionminers_real': 5, 'motionminers_flw': 5}
    num_filters = {'mocap': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64},
                   'mbientlab': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64},
                   'virtual': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64},
                   'mocap_half': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64},
                   'virtual_quarter': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64},
                   'mocap_quarter': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64},
                   'mbientlab_50_p': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64},
                   'mbientlab_10_p': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64},
                   'mbientlab_50_r': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64},
                   'mbientlab_10_r': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64},
                   'mbientlab_quarter': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64},
                   'motionminers_real': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64},
                   'motionminers_flw': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_tpp': 64, 'cnn_imu_tpp': 64}}

    freeze_options = [False, True]

    # Evolution
    evolution_iter = 10000

    # Results will be stored in different folders according to the dataset and network
    # This as a sort of organisation for tracking the experiments
    # dataset/network/output/MLP_type/input_shape/
    # dataset/network/output/MLP_type/input_shape/experiment
    # dataset/network/output/MLP_type/input_shape/experiment/plots
    # dataset/network/output/MLP_type/input_shape/final
    # dataset/network/output/MLP_type/input_shape/final/plots
    # dataset/network/output/MLP_type/input_shape/fine_tuning
    # dataset/network/output/MLP_type/input_shape/fine_tuning/plots

    # User gotta take care of creating these folders, or storing the results in a different way

    reshape_input = reshape_input
    if reshape_input:
        reshape_folder = "reshape"
    else:
        reshape_folder = "noreshape"

    if fully_convolutional:
        fully_convolutional = "FCN"
    else:
        fully_convolutional = "FC"

    if pooling == 1 or pooling == 2:
        pooling_pooling = "pooling"
    else:
        pooling_pooling = "nopooling"

    if output[output_idx] == 'softmax':
        labeltype = "class"
        folder_base = "/data2/"
    elif output[output_idx] == 'attribute':
        labeltype = "attributes"
        folder_base = "/data2/"
    elif output[output_idx] == 'identity':
        labeltype = "identity"
        folder_base = "/data2/"


    # Folder
    if usage_modus[usage_modus_idx] == 'train':
        folder_exp = folder_base + 'fmoya/HAR/pytorch/' + dataset[dataset_idx] + '/' + \
                     network[network_idx] + '/' + output[output_idx] + '/' + pooling_pooling + '/' + \
                     fully_convolutional + '/' + reshape_folder + '/' + 'experiment/'
        folder_exp_base_fine_tuning = folder_base + 'fmoya/HAR/pytorch/' + dataset[dataset_fine_tuning_idx] + '/' + \
                                      network[network_idx] + '/' + output[output_idx] + '/' + pooling_pooling + '/' + \
                                      fully_convolutional + '/' + reshape_folder + '/' + 'final/'
    elif usage_modus[usage_modus_idx] == 'test':
        folder_exp = folder_base + 'fmoya/HAR/pytorch/' + dataset[dataset_idx] + '/' + \
                     network[network_idx] + '/' + output[output_idx] + '/' + pooling_pooling + '/' + \
                     fully_convolutional + \
                     '/' + reshape_folder + '/' + 'final/'
        folder_exp_base_fine_tuning = folder_base + 'fmoya/HAR/pytorch/' + dataset[dataset_fine_tuning_idx] + '/' + \
                                      network[network_idx] + '/' + output[output_idx] + '/' + pooling_pooling + '/' + \
                                      fully_convolutional + '/' + reshape_folder + '/' + 'final/'
    elif usage_modus[usage_modus_idx] == 'train_final':
        folder_exp = folder_base + 'fmoya/HAR/pytorch/' + dataset[dataset_idx] + '/' + \
                     network[network_idx] + '/' + output[output_idx] + '/' + pooling_pooling + '/' + \
                     fully_convolutional + '/' + reshape_folder + '/' + 'final/'
        folder_exp_base_fine_tuning = folder_base + 'fmoya/HAR/pytorch/' + dataset[dataset_fine_tuning_idx] + '/' + \
                                      network[network_idx] + '/' + output[output_idx] + '/' + pooling_pooling + '/' + \
                                      fully_convolutional + '/' + reshape_folder + '/' + 'final/'
    elif usage_modus[usage_modus_idx] == 'fine_tuning':
        folder_exp = folder_base + 'fmoya/HAR/pytorch/' + dataset[dataset_idx] + '/' + \
                     network[network_idx] + '/' + output[output_idx] + '/' + pooling_pooling + '/' + \
                     fully_convolutional + '/' + reshape_folder + '/' + 'fine_tuning/'
        folder_exp_base_fine_tuning = folder_base + 'fmoya/HAR/pytorch/' + dataset[dataset_fine_tuning_idx] + '/' + \
                                      network[network_idx] + '/' + output[output_idx] + '/' + pooling_pooling + '/' + \
                                      fully_convolutional + '/' + reshape_folder + '/' + 'final/'
    else:
        raise ("Error: Not selected fine tuning option")

    dataset_root = {'mocap': folder_base + 'fmoya/HAR/datasets/MoCap_dataset/',
                    'mbientlab': folder_base + 'fmoya/HAR/datasets/mbientlab/',
                    'virtual': folder_base + 'fmoya/HAR/datasets/Virtual_IMUs/',
                    'mocap_half': folder_base + 'fmoya/HAR/datasets/MoCap_dataset_half_freq/',
                    'virtual_quarter': folder_base + 'fmoya/HAR/datasets/Virtual_IMUs/',
                    'mocap_quarter': folder_base + 'fmoya/HAR/datasets/MoCap_dataset_half_freq/',
                    'mbientlab_50_p': folder_base + 'fmoya/HAR/datasets/mbientlab_50_persons/',
                    'mbientlab_10_p': folder_base + 'fmoya/HAR/datasets/mbientlab_10_persons/',
                    'mbientlab_50_r': folder_base + 'fmoya/HAR/datasets/mbientlab_50_recordings/',
                    'mbientlab_10_r': folder_base + 'fmoya/HAR/datasets/mbientlab_10_recordings/',
                    'mbientlab_quarter': folder_base + 'fmoya/HAR/datasets/mbientlab/',
                    'motionminers_real': folder_base + 'fmoya/HAR/datasets/motionminers_real/',
                    'motionminers_flw': folder_base + 'fmoya/HAR/datasets/motionminers_flw/'}

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    GPU = 0

    # Labels position on the segmented window
    label_pos = {0: 'middle', 1: 'mode', 2: 'end'}

    percentages_names = ["001", "002", "005", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    percentages_dataset = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    #train_show_value = num_tr_inputs[dataset[dataset_idx]] * percentages_dataset[percentage_idx]
    train_show_value = num_tr_inputs[dataset[dataset_idx]] / \
                       batch_size_train[network[network_idx]][dataset[dataset_idx]]
    if dataset[dataset_idx] == "mbientlab" or dataset[dataset_idx] == "motionminers_real":
        train_show = {'cnn': int(train_show_value / 50), 'lstm': 50, 'cnn_imu': int(train_show_value / 50),
                      'cnn_tpp': int(train_show_value / 50), 'cnn_imu_tpp': int(train_show_value / 50)}
        valid_show = {'cnn': int(train_show_value / 10), 'lstm': 10, 'cnn_imu': int(train_show_value / 10),
                      'cnn_tpp': int(train_show_value / 10), 'cnn_imu_tpp': int(train_show_value / 10)}
    elif dataset[dataset_idx] == "mocap":
        train_show = {'cnn': int(train_show_value / 100), 'lstm': 100, 'cnn_imu': int(train_show_value / 100),
                      'cnn_tpp': int(train_show_value / 100), 'cnn_imu_tpp': int(train_show_value / 100)}
        valid_show = {'cnn': int(train_show_value / 20), 'lstm': 50, 'cnn_imu': int(train_show_value / 20),
                      'cnn_tpp': int(train_show_value / 20), 'cnn_imu_tpp': int(train_show_value / 20)}
    else:
        train_show = {'cnn': int(train_show_value / 100), 'lstm': 100, 'cnn_imu': int(train_show_value / 100),
                      'cnn_tpp': int(train_show_value / 100), 'cnn_imu_tpp': int(train_show_value / 100)}
        valid_show = {'cnn': int(train_show_value / 50), 'lstm': 50, 'cnn_imu': int(train_show_value / 50),
                      'cnn_tpp': int(train_show_value / 50), 'cnn_imu_tpp': int(train_show_value / 50)}

    now = datetime.datetime.now()

    configuration = {'dataset': dataset[dataset_idx],
                     'dataset_finetuning': dataset[dataset_fine_tuning_idx],
                     'network': network[network_idx],
                     'output': output[output_idx],
                     'num_filters': num_filters[dataset[dataset_idx]][network[network_idx]],
                     'filter_size': filter_size[dataset[dataset_idx]],
                     'lr': lr[dataset[dataset_idx]][network[network_idx]] * lr_mult,
                     'epochs': epochs[dataset[dataset_idx]][network[network_idx]][output[output_idx]] * epoch_mult,
                     'evolution_iter': evolution_iter,
                     'train_show': train_show[network[network_idx]],
                     'valid_show': valid_show[network[network_idx]],
                     'plotting': plotting,
                     'usage_modus': usage_modus[usage_modus_idx],
                     'folder_exp': folder_exp,
                     'folder_exp_base_fine_tuning': folder_exp_base_fine_tuning,
                     'use_maxout': use_maxout[network[network_idx]],
                     'balancing': balancing[dataset[dataset_idx]],
                     'GPU': GPU,
                     'division_epochs': division_epochs[dataset[dataset_idx]],
                     'NB_sensor_channels': NB_sensor_channels[dataset[dataset_idx]],
                     'sliding_window_length': sliding_window_length[dataset[dataset_idx]],
                     'sliding_window_step': sliding_window_step[dataset[dataset_idx]],
                     'num_attributes': num_attributes[dataset[dataset_idx]],
                     'batch_size_train': batch_size_train[network[network_idx]][dataset[dataset_idx]],
                     'batch_size_val': batch_size_val[network[network_idx]][dataset[dataset_idx]],
                     'num_tr_inputs': num_tr_inputs[dataset[dataset_idx]],
                     'num_classes': num_classes[dataset[dataset_idx]],
                     'label_pos': label_pos[2],
                     'file_suffix': 'results_yy{}mm{}dd{:02d}hh{:02d}mm{:02d}.xml'.format(now.year,
                                                                                          now.month,
                                                                                          now.day,
                                                                                          now.hour,
                                                                                          now.minute),
                     'dataset_root': dataset_root[dataset[dataset_idx]],
                     'accumulation_steps': accumulation_steps[dataset[dataset_idx]],
                     'reshape_input': reshape_input,
                     'name_counter': name_counter,
                     'freeze_options': freeze_options[freeze],
                     'percentages_names': percentages_names[percentage_idx],
                     'fully_convolutional': fully_convolutional,
                     'sacred': sacred,
                     'labeltype': labeltype,
                     'pooling': pooling}

    return configuration

def setup_experiment_logger(logging_level=logging.DEBUG, filename=None):
    # set up the logging
    logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'

    if filename != None:
        logging.basicConfig(filename=filename, level=logging.DEBUG,
                            format=logging_format,
                            filemode='w')
    else:
        logging.basicConfig(level=logging_level,
                            format=logging_format,
                            filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger

    if logging.getLogger('').hasHandlers():
        logging.getLogger('').handlers.clear()

    logging.getLogger('').addHandler(console)

    return


@ex.config
def my_config():
    config = configuration(dataset_idx=11,
                           network_idx=0,
                           output_idx=0,
                           usage_modus_idx=1,
                           dataset_fine_tuning_idx=0,
                           reshape_input=False,
                           learning_rates_idx=0,
                           name_counter=0,
                           freeze=0,
                           percentage_idx=12,
                           fully_convolutional=False,
                           pooling=0)
    dataset = config["dataset"]
    network = config["network"]
    output = config["output"]
    reshape_input = config["reshape_input"]
    usageModus = config["usage_modus"]
    dataset_finetuning = config["dataset_finetuning"]
    pooling = config["pooling"]
    lr = config["lr"]



@ex.capture
def run(config, dataset, network, output, usageModus):
    setup_experiment_logger(logging_level=logging.DEBUG,
                            filename=config['folder_exp'] + "logger.txt")

    logging.info('Finished')
    logging.info('Dataset {} Network {} Output {} Modus {}'.format(dataset, network, output, usageModus))

    modus = Modus_Selecter(config, ex)

    # Starting process
    modus.net_modus()

    print("Done")


@ex.automain
def main():

    #Setting the same RNG seed
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

    print("Python Platform {}".format(platform.python_version()))


    run()

    print("Done")