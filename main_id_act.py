# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 09:02:03 2021

@author: nilah
"""

from __future__ import print_function
import os
import logging
import torch
import numpy as np
import random

import platform
from modus_selecter_act import Modus_Selecter

import datetime

from sacred import Experiment
#from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

ex= Experiment('id_count_attr')

ex.observers.append(MongoObserver.create(url='curtiz',
                                         db_name='nnair_sacred',
                                         username='nnair',
                                         password='Germany2018',
                                         authSource='admin',
                                         authMechanism='SCRAM-SHA-1'))

def configuration(dataset_idx, network_idx, output_idx, usage_modus_idx=0, dataset_fine_tuning_idx=0,
                  reshape_input=False, learning_rates_idx=0, name_counter=0, freeze=0, percentage_idx=0,
                  fully_convolutional=False, sacred=True):
    
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
    @return configuration: dict with all the configurations
    """
    # Flags
    plotting = True

    # Options
    dataset = {0: 'mocap', 1: 'mbientlab', 2: 'motionminers_flw'}
    network = {0: 'cnn', 1: 'lstm', 2: 'cnn_imu'}
    output = {0: 'softmax', 1: 'attribute'}
    usage_modus = {0: 'train', 1: 'test', 2: 'fine_tuning', 3: 'train_final'}

    # Dataset Hyperparameters
    NB_sensor_channels = {'mocap': 126, 'mbientlab': 30,'motionminers_flw': 27}
    sliding_window_length = {'mocap': 100, 'mbientlab': 100, 'motionminers_flw': 100}
    sliding_window_step = {'mocap': 12, 'mbientlab': 12, 'motionminers_flw': 12}
    
    num_attributes = {'mocap': 4, 'mbientlab': 4, 'motionminers_flw': 4}
    #num_attributes = {'mocap': 11, 'mbientlab': 11, 'motionminers_flw': 11}
    
    num_tr_inputs = {'mocap': 247702, 'mbientlab': 91399, 'motionminers_flw': 93712}
    
    num_classes = {'mocap': 7, 'mbientlab': 7, 'motionminers_flw': 7}
    
    # It was thought to have different LR per dataset, but experimentally have worked the next three
    # Learning rate
    learning_rates = [0.0001, 0.00001, 0.000001, 0.01]
    lr = {'mocap': {'cnn': learning_rates[learning_rates_idx],
                    'lstm': learning_rates[learning_rates_idx],
                    'cnn_imu': learning_rates[learning_rates_idx]},
          'mbientlab': {'cnn': learning_rates[learning_rates_idx],
                        'lstm': learning_rates[learning_rates_idx],
                        'cnn_imu': learning_rates[learning_rates_idx]},
          'motionminers_flw': {'cnn': learning_rates[learning_rates_idx],
                                'lstm': learning_rates[learning_rates_idx],
                                'cnn_imu': learning_rates[learning_rates_idx]}
          }
    lr_mult = 1.0

    # Maxout
    #use_maxout = {'cnn': False, 'lstm': False, 'cnn_imu': False}

    # Balacing the proportion of classes into the dataset dataset
    # This will be deprecated
    #balancing = {'mocap': False, 'mbientlab': False, 'virtual': False, 'mocap_half': False, 'virtual_quarter': False,
    #             'mocap_quarter': False, 'mbientlab_50_p': False, 'mbientlab_10_p': False, 'mbientlab_50_r': False,
    #             'mbientlab_10_r': False, 'mbientlab_quarter': False, 'motionminers_real': False,
    #             'motionminers_flw': False}
    
    if usage_modus[usage_modus_idx] == 'train_final' or usage_modus[usage_modus_idx] == 'fine_tuning':
        epoch_mult = 2
    else:
        epoch_mult = 1

   # Number of epochs depending of the dataset and network
    epochs = {'mocap': {'cnn': {'softmax': 10, 'attribute': 10},
                        'lstm': {'softmax': 10, 'attribute': 10},
                        'cnn_imu': {'softmax': 1, 'attribute': 10}},
              'mbientlab': {'cnn': {'softmax': 10, 'attribute': 10},
                            'lstm': {'softmax': 10, 'attribute': 10},
                            'cnn_imu': {'softmax': 10, 'attribute': 10}},
              'motionminers_flw': {'cnn': {'softmax': 10, 'attribute': 10},
                                   'lstm': {'softmax': 10, 'attribute': 10},
                                   'cnn_imu': {'softmax': 10, 'attribute': 10}}
              } 
   #division_epochs = {'mocap': 2, 'mbientlab': 1, 'motionminers_flw': 1}

    # Batch size
    batch_size_train = {
        'cnn': {'mocap': 100, 'mbientlab': 100, 'motionminers_flw': 100},
        'lstm': {'mocap': 100, 'mbientlab': 100, 'motionminers_flw': 100},
        'cnn_imu': {'mocap': 100, 'mbientlab':100, 'motionminers_flw': 100}}

    batch_size_val = {'cnn': {'mocap': 100, 'mbientlab': 100, 'motionminers_flw': 100},
                      'lstm': {'mocap': 100, 'mbientlab': 100, 'motionminers_flw': 100},
                      'cnn_imu': {'mocap': 100, 'mbientlab':100, 'motionminers_flw': 100}}
    
     # Number of iterations for accumulating the gradients
    accumulation_steps = {'mocap': 4, 'mbientlab': 4, 'motionminers_flw': 4}

    # Filters
    filter_size = {'mocap': 5, 'mbientlab': 5, 'motionminers_flw': 5}
    num_filters = {'mocap': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64},
                   'mbientlab': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64},
                   'motionminers_flw': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64}}

    freeze_options = [False, True]
    #evolution_iter = 10000
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

    if output[output_idx] == 'softmax':
        labeltype = "class"
        #folder_base = "/data/nnair/output/softmax/clean/"
        folder_base = "/data/nnair/trial/"
    elif output[output_idx] == 'attribute':
        labeltype = "attributes"
        folder_base = "/data/nnair/output/attributes/"
        
    print("folderbase selected")
    print(folder_base)

##################################Check this again###############################################
    
    # Folder
    if usage_modus[usage_modus_idx] == 'train':
        '''
        folder_exp = folder_base + dataset[dataset_idx] + '/' + \
                     network[network_idx] + '/' + fully_convolutional \
                     + '/' + reshape_folder +'/' + 'experiment2/'
        '''
        folder_exp = folder_base + 'experiment/'
        print(folder_exp)
        '''
        folder_exp_base_fine_tuning = folder_base + dataset[dataset_fine_tuning_idx] + '/' + \
                                      network[network_idx] + '/' + fully_convolutional \
                                      + '/' + 'train_final/'
        '''
    elif usage_modus[usage_modus_idx] == 'test':
        folder_exp = folder_base + dataset[dataset_idx] + '/' + \
                     network[network_idx] + '/' + fully_convolutional \
                     + '/' + reshape_folder +'/' + 'test_final/'
        print(folder_exp)
        '''
        folder_exp_base_fine_tuning = folder_base + dataset[dataset_fine_tuning_idx] + '/' + \
                                      network[network_idx] +  fully_convolutional + \
                                      '/' + 'final/'
        '''
    elif usage_modus[usage_modus_idx] == 'train_final':
        folder_exp = folder_base + dataset[dataset_idx] + '/' + \
                     network[network_idx] + '/' + fully_convolutional +\
                     '/' + reshape_folder + '/' + 'train_final/'
        print(folder_exp)
        '''
        folder_exp_base_fine_tuning = folder_base + dataset[dataset_fine_tuning_idx] + '/' + \
                                      network[network_idx] + '/' + fully_convolutional + \
                                      '/' + 'train_final/'
        '''
    elif usage_modus[usage_modus_idx] == 'fine_tuning':
        folder_exp = folder_base + dataset[dataset_idx] + '/' + \
                     network[network_idx] + '/' + fully_convolutional + \
                     + '/' + reshape_folder +'/' + 'fine_tuning/'
        print(folder_exp)
        '''
        folder_exp_base_fine_tuning = folder_base + dataset[dataset_fine_tuning_idx] + '/' + \
                                      network[network_idx] + '/' + fully_convolutional + \
                                      '/' + 'final/'
        '''
    else:
        raise ("Error: Not selected fine tuning option")
    
################################################################################################################################3

    # Paths are given according to the ones created in *preprocessing.py for the datasets
    
       
    dataset_root = {'mocap': '/data/nnair/output/type1/mocap/downsampled/',
                    'mbientlab': '/data/nnair/output/type1/imu/',
                    'motionminers_flw': '/data/nnair/output/type1/momin/'}
  
    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    GPU = 0
   
    # Labels position on the segmented window
    label_pos = {0: 'middle', 1: 'mode', 2: 'end'}
    
    train_show_value = num_tr_inputs[dataset[dataset_idx]] / \
                       batch_size_train[network[network_idx]][dataset[dataset_idx]]
    if dataset[dataset_idx] == "mbientlab" or dataset[dataset_idx] == "motionminers_real":
        train_show = {'cnn': int(train_show_value / 50), 'lstm': 50, 'cnn_imu': int(train_show_value / 50)}
        valid_show = {'cnn': int(train_show_value / 10), 'lstm': 10, 'cnn_imu': int(train_show_value / 10)}
    elif dataset[dataset_idx] == "mocap":
        train_show = {'cnn': int(train_show_value / 100), 'lstm': 100, 'cnn_imu': int(train_show_value / 100)}
        valid_show = {'cnn': int(train_show_value / 20), 'lstm': 50, 'cnn_imu': int(train_show_value / 20)}
    else:
        train_show = {'cnn': int(train_show_value / 100), 'lstm': 100, 'cnn_imu': int(train_show_value / 100)}
        valid_show = {'cnn': int(train_show_value / 50), 'lstm': 50, 'cnn_imu': int(train_show_value / 50)}
    
    
    now = datetime.datetime.now()
    
    configuration = {'dataset': dataset[dataset_idx],
                     #'dataset_finetuning': dataset[dataset_fine_tuning_idx],
                     'network': network[network_idx],
                     'output': output[output_idx],
                     'num_filters': num_filters[dataset[dataset_idx]][network[network_idx]],
                     'filter_size': filter_size[dataset[dataset_idx]],
                     'lr': lr[dataset[dataset_idx]][network[network_idx]] * lr_mult,
                     'epochs': epochs[dataset[dataset_idx]][network[network_idx]][output[output_idx]] * epoch_mult,
                     'train_show': train_show[network[network_idx]],
                     'valid_show': valid_show[network[network_idx]],
                     'plotting': plotting,
                     'usage_modus': usage_modus[usage_modus_idx],
                     'folder_exp': folder_exp,
                     #'folder_exp_base_fine_tuning': folder_exp_base_fine_tuning,
                     #'balancing': balancing[dataset[dataset_idx]],
                     'GPU': GPU,
                     #'division_epochs': division_epochs[dataset[dataset_idx]],
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
                     #'percentages_names': percentages_names[percentage_idx],
                     'fully_convolutional': fully_convolutional,
                     'sacred': sacred,
                     'labeltype': labeltype}

    return configuration

def setup_experiment_logger(logging_level=logging.DEBUG, filename=None):
    print("setup logger began")
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
    print("configuration function began")
    config = configuration(dataset_idx=0,
                           network_idx=2,
                           output_idx=0,
                           usage_modus_idx=0,
                           #dataset_fine_tuning_idx=0,
                           reshape_input=False,
                           learning_rates_idx=0,
                           name_counter=0,
                           freeze=0,
                           fully_convolutional=False,
                           #percentage_idx=12,
                           #pooling=0
                           )
    
    dataset = config["dataset"]
    network = config["network"]
    output = config["output"]
    reshape_input = config["reshape_input"]
    usageModus = config["usage_modus"]
    #dataset_finetuning = config["dataset_finetuning"]
    #pooling = config["pooling"]
    lr = config["lr"]
    bsize = config["batch_size_train"]
    
@ex.capture
def run(config, dataset, network, output, usageModus):
   
    file_name='/data/nnair/output/avg/'
   
    file_name='/data/nnair/output/avg/'+'logger.txt'
    
    setup_experiment_logger(logging_level=logging.DEBUG,filename=file_name)

    logging.info('Finished')
    logging.info('Dataset {} Network {} Output {} Modus {}'.format(dataset, network, output, usageModus))

    modus = Modus_Selecter(config, ex)

    # Starting process
    modus.net_modus()

    print("Done")


@ex.automain
def main():
    print("main began")
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

    print("Python  {}".format(platform.python_version()))


    run()

    print("Done")

'''    
def main():
    """
    Run experiment for a certain set of parameters

    User is welcome to revise in detatil the configuration function
    for more information about all of possible configurations for the experiments

    """
    dataset_idx = [1]
    network_idx = [2]
    reshape_input = [True]
    output_idxs = [0]
    lrs = [0, 1, 2]
    #dataset_ft_idx = [0,1,2,3]
    counter_exp = 0
    freeze = [0]
    #percentages = [12]
    for dts in range(len(dataset_idx)):
        for nt in range(len(network_idx)):
            for opt in output_idxs:
                #for dft in dataset_ft_idx:
                    #for pr in percentages:
                        for rsi in range(len(reshape_input)):
                            for fr in freeze:
                                for lr in lrs:
                                    config = configuration(dataset_idx=dataset_idx[dts],
                                                           network_idx=network_idx[nt],
                                                           output_idx=opt,
                                                           usage_modus_idx=5,
                                                           #dataset_fine_tuning_idx=dft,
                                                           reshape_input=reshape_input[rsi],
                                                           learning_rates_idx=lr,
                                                           name_counter=counter_exp,
                                                           freeze=fr,
                                                           #percentage_idx=pr,
                                                           fully_convolutional=False)

                                    setup_experiment_logger(logging_level=logging.DEBUG,
                                                            filename=config['/data/nnair/output/'] + "logger.txt")

                                    logging.info('Finished')

                                    modus = Modus_Selecter(config)

                                    # Starting process
                                    modus.net_modus()
                                    counter_exp += 1


    return


if __name__ == '__main__':

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
    
    main()

    print("Done")
'''