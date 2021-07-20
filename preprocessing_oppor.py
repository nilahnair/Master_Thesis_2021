# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 07:49:13 2021

@author: nilah
adapted from: 'fjordonez'
"""

import os
import zipfile
import argparse
import numpy as np
import pickle

from io import BytesIO
from pandas import Series

import numpy as np
import sys
#import torch.utils.data as data
import logging
from sliding_window_dat import sliding_window
#from resampling import Resampling



# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113
NUM_ACT_CLASSES= 5
NUM_CLASSES =4

ws = 100
ss = 12

OPPORTUNITY_DATA_FILES = ['OpportunityUCIDataset/dataset/S1-Drill.dat', #0
                          'OpportunityUCIDataset/dataset/S1-ADL1.dat',  #1
                          'OpportunityUCIDataset/dataset/S1-ADL2.dat',  #2
                          'OpportunityUCIDataset/dataset/S1-ADL3.dat',  #3
                          'OpportunityUCIDataset/dataset/S1-ADL4.dat',  #4
                          'OpportunityUCIDataset/dataset/S1-ADL5.dat',  #5
                          'OpportunityUCIDataset/dataset/S2-Drill.dat', #6
                          'OpportunityUCIDataset/dataset/S2-ADL1.dat',  #7
                          'OpportunityUCIDataset/dataset/S2-ADL2.dat',  #8
                          'OpportunityUCIDataset/dataset/S3-Drill.dat', #9
                          'OpportunityUCIDataset/dataset/S3-ADL1.dat',  #10
                          'OpportunityUCIDataset/dataset/S3-ADL2.dat',  #11
                          'OpportunityUCIDataset/dataset/S2-ADL3.dat',  #12
                          'OpportunityUCIDataset/dataset/S3-ADL3.dat',  #13
                          'OpportunityUCIDataset/dataset/S2-ADL4.dat',  #14
                          'OpportunityUCIDataset/dataset/S2-ADL5.dat',  #15
                          'OpportunityUCIDataset/dataset/S3-ADL4.dat',  #16
                          'OpportunityUCIDataset/dataset/S3-ADL5.dat'   #17
                          ]
persons = ["S1", "S2", "S3", "S4"]
ID ={"S1": 0, "S2": 1, "S3": 2, "S4": 3,}
train_data_files = ['/vol/actrec/Opportunity/dataset/S1-ADL1.dat', #0
                    '/vol/actrec/Opportunity/dataset/S1-ADL2.dat', #1
                    '/vol/actrec/Opportunity/dataset/S1-ADL3.dat', #2
                    '/vol/actrec/Opportunity/dataset/S2-ADL1.dat', #3
                    '/vol/actrec/Opportunity/dataset/S2-ADL2.dat', #4
                    '/vol/actrec/Opportunity/dataset/S2-ADL3.dat', #5
                    '/vol/actrec/Opportunity/dataset/S3-ADL1.dat', #6
                    '/vol/actrec/Opportunity/dataset/S3-ADL2.dat', #7
                    '/vol/actrec/Opportunity/dataset/S3-ADL3.dat', #8
                    '/vol/actrec/Opportunity/dataset/S4-ADL1.dat', #9
                    '/vol/actrec/Opportunity/dataset/S4-ADL2.dat', #10
                    '/vol/actrec/Opportunity/dataset/S4-ADL3.dat'  #11
                    ]
val_data_files = ['/vol/actrec/Opportunity/dataset/S1-ADL4.dat', #0
                  '/vol/actrec/Opportunity/dataset/S2-ADL4.dat', #1
                  '/vol/actrec/Opportunity/dataset/S3-ADL4.dat', #2
                  '/vol/actrec/Opportunity/dataset/S4-ADL4.dat'  #3
                 ]
test_data_files = ['/vol/actrec/Opportunity/dataset/S1-ADL5.dat', #0
                  '/vol/actrec/Opportunity/dataset/S2-ADL5.dat',  #1
                  '/vol/actrec/Opportunity/dataset/S3-ADL5.dat',  #2
                  '/vol/actrec/Opportunity/dataset/S4-ADL5.dat'   #3
                 ]

# Hardcoded thresholds to define global maximums and minimums for every one of the 113 sensor channels employed in the
# OPPORTUNITY challenge
NORM_MAX_THRESHOLDS = [3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
                       10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
                       200,    5000,   5000,   5000,   5000,   5000,   5000,   10000,  10000,
                       10000,  10000,  10000,  10000,  250, ]

NORM_MIN_THRESHOLDS = [-3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                       -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
                       -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -10000, -10000,
                       -10000, -10000, -10000, -10000, -250, ]

def select_columns_opp(data):
    """Selection of the 113 columns employed in the OPPORTUNITY challenge

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: numpy integer matrix
        Selection of features
    """

    #                     included-excluded
    features_delete = np.arange(46, 50)
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])
    return np.delete(data, features_delete, 1)

def opp_sliding_window(data_x, data_y, data_z, label_pos_end=True):
    '''
    print("check1")
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    print(data_x.shape)
    
    #return data_x.astype(np.float32), data_y.astype(np.uint8)
    return data_x.astype(np.float32)
    '''
    print("Sliding window: Creating windows {} with step {}".format(ws, ss))

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    print(data_x.shape)
    # Label from the end
    if label_pos_end:
        print("check 1")
        data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
        data_z = np.asarray([[i[-1]] for i in sliding_window(data_z, ws, ss)])
    else:
        if False:
            # Label from the middle
            # not used in experiments
            print("check 2")
            data_y_labels = np.asarray(
                [[i[i.shape[0] // 2]] for i in sliding_window(data_y, ws, ss)])
            data_z_labels = np.asarray(
                [[i[i.shape[0] // 2]] for i in sliding_window(data_z, ws, ss)])
        else:
            # Label according to mode
            try:
                print("check 3")
                data_y_labels = []
                data_z_labels = []
                for sw in sliding_window(data_y, ws, ss):
        
                    count_l = np.bincount(sw.astype(int), minlength=NUM_ACT_CLASSES)
                    idy = np.argmax(count_l)
                    data_y_labels.append(idy)
                data_y_labels = np.asarray(data_y_labels)
                for sz in sliding_window(data_z, ws, ss):
                    count_l = np.bincount(sz.astype(int), minlength=NUM_CLASSES)
                    idy = np.argmax(count_l)
                    data_z_labels.append(idy)
                data_z_labels = np.asarray(data_z_labels)


            except:
                print("Sliding window: error with the counting {}".format(count_l))
                print("Sliding window: error with the counting {}".format(idy))
                return np.Inf

            # All labels per window
            data_y_all = np.asarray([i[:] for i in sliding_window(data_y, ws, ss)])
            print(data_y_all.shape)
            data_z_all = np.asarray([i[:] for i in sliding_window(data_z, ws, ss)])
            print(data_z_all.shape)
            
    print("daya_y_labels")
    print(data_y_labels.shape)
    print("daya_y_all")
    print(data_y_all.shape)
    print("daya_z_labels")
    print(data_z_labels.shape)
    print("daya_z_all")
    print(data_z_all.shape)

    return data_x.astype(np.float32), data_y_labels.astype(np.uint8), data_y_all.astype(np.uint8), data_z_labels.astype(np.uint8), data_z_all.astype(np.uint8)

def normalize(data, max_list, min_list):
    """Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :param max_list: numpy integer array
        Array containing maximums values for every one of the 113 sensor channels
    :param min_list: numpy integer array
        Array containing minimum values for every one of the 113 sensor channels
    :return:
        Normalized sensor data
    """
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i]-min_list[i])/diffs[i]
    #     Checking the boundaries
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data

def divide_x_y(data, label):
    """Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """

    data_x = data[:, 1:114]
    if label not in ['locomotion', 'gestures']:
            raise RuntimeError("Invalid label: '%s'" % label)
    if label == 'locomotion':
        print("Locomotion")
        data_y = data[:, 114]  # Locomotion label
    elif label == 'gestures':
        print("Gestures")
        data_y = data[:, 115]  # Gestures label

    return data_x, data_y

def adjust_idx_labels(data_y, label):
    """Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    """

    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y

def process_dataset_file(data, label):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """

    # Select correct columns
    data = select_columns_opp(data)

    # Colums are segmentd into features and labels
    data_x, data_y =  divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    # All sensor channels are normalized
    data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

    return data_x, data_y


def generate_data(target_filename, label, datatype):
    """
    :param target_filename: string
        Processed file
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    :param datatype: string ['train', 'val', 'test']
        for choosing the sub categories for saving files
    """
    if datatype == 'train':
        print("initialising train variables")
        X_train = np.empty((0, NB_SENSOR_CHANNELS))
        act_train = np.empty((0))
        id_train = np.empty((0))
        data_files = train_data_files
    elif datatype == 'val':
        print("initialising val variables")
        X_val = np.empty((0, NB_SENSOR_CHANNELS))
        act_val = np.empty((0))
        id_val = np.empty((0))
        data_files = val_data_files
    elif datatype == 'test':
        print("initialising test variables")
        X_test = np.empty((0, NB_SENSOR_CHANNELS))
        act_test = np.empty((0))
        id_test = np.empty((0))
        data_files = test_data_files
    counter_files = 0
    
    print('Processing dataset files ...')
    for d in data_files:
        try:
            data = np.loadtxt(d)
            print('file {0}'.format(d))
            x, y = process_dataset_file(data, label)
            print(x.shape)
            print(y.shape)
            if datatype == 'train':
                print("concatenating train values")
                X_train = np.vstack((X_train, x))
                act_train = np.concatenate([act_train, y])
                if counter_files >=0 and counter_files<3:
                    id= np.full(y.shape, 0)
                elif counter_files>=3 and counter_files <6:
                    id= np.full(y.shape, 1)
                elif counter_files>=6 and counter_files <9:
                    id = np.full(y.shape, 2)
                elif counter_files >=9:
                    id = np.full(y.shape,3)
                id_train = np.concatenate([id_train, id])
            elif datatype == 'val':
                print("concatenating val values")
                X_val = np.vstack((X_val, x))
                act_val = np.concatenate([act_val, y])
                id_val = np.concatenate([id_val, np.full(y.shape,counter_files)])
            elif datatype == 'test':
                print("concatenating test values")
                X_test = np.vstack((X_test, x))
                act_test = np.concatenate([act_test, y])
                id_test = np.concatenate([id_test, np.full(y.shape,counter_files)])
            
        except KeyError:
            print('ERROR: Did not find {0} in zip file'.format(d))
            
        counter_files += 1 
    try:    
        print("performing sliding window")
        if datatype == 'train':
            X, act, act_all, labelid, labelid_all = opp_sliding_window(X_train, act_train, id_train, label_pos_end = False)
        elif datatype == 'val':
            X, act, act_all, labelid, labelid_all = opp_sliding_window(X_val, act_val, id_val, label_pos_end = False)
        elif datatype == 'test':
            X, act, act_all, labelid, labelid_all = opp_sliding_window(X_test, act_test, id_test, label_pos_end = False)
    except:
        print("error in sliding window")
        
    try:
        counter_seq = 0
        print("window extraction begining")
        print("target file name")
        print(target_filename())
        for f in range(X.shape[0]):
            try:
                sys.stdout.write('\r' + 'Creating sequence file '
                                 'number {} with id {}'.format(f, counter_seq))
                sys.stdout.flush()

                # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                seq = np.reshape(X[f], newshape = (1, X.shape[1], X.shape[2]))
                seq = np.require(seq, dtype=np.float)
                print("seq")
                print(seq.shape)
                print("act_label")
                print(act[f].shape)
                print("act_all")
                print(act_all[f].shape)
                print("id")
                print(labelid[f].shape)
                print("id_all")
                print(labelid_all[f].shape)
                # Storing the sequences
                #obj = {"data": seq, "label": labelid}
                obj = {"data": seq, "act_label": act[f], "act_labels_all": act_all[f], "label": labelid[f]}
                
                f = open(os.path.join(target_filename, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()

                counter_seq += 1
            except:
                raise('\nError adding the seq')
    except:
        print("error in saving")
    # Dataset is segmented into train and test
    #nb_training_samples = 557963
    # The first 18 OPPORTUNITY data files define the traning dataset, comprising 557963 samples
    #X_train, y_train = data_x[:nb_training_samples,:], data_y[:nb_training_samples]
    #X_test, y_test = data_x[nb_training_samples:,:], data_y[nb_training_samples:]

    #print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape))
    '''
    if datatype == 'train':
        obj = [(X_train, act_train, id_ ), (X_val, y_val), (X_test, y_test)]
    f = open(os.path.join('/data2/fmoya/HAR/opportunity', target_filename), 'wb')
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    '''
    
    
if __name__ == '__main__':
    
    base_directory = '/data/nnair/oppor/locomotions/inputs/'
        
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    l= 'locomotion'
    generate_data(target_filename=data_dir_train, label=l, datatype='train')
    generate_data(target_filename=data_dir_val, label=l, datatype='val')
    generate_data(target_filename=data_dir_test, label=l, datatype='test')
    
    ##############check this parts neccesity
    '''
    generate_CSV(base_directory + "train.csv", data_dir_train)
    generate_CSV(base_directory + "val.csv", data_dir_val)
    generate_CSV(base_directory + "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)
    '''
    print("Done")