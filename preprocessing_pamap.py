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
NB_SENSOR_CHANNELS = 40
NUM_ACT_CLASSES= 12
NUM_CLASSES =9

ws = 100
ss = 22

PAMAP2_DATA_FILES = ['/vol/actrec/PAMAP/PAMAP2_Dataset/Protocol/subject101.dat', #0
                     '/vol/actrec/PAMAP/PAMAP2_Dataset/Protocol/subject102.dat', #1
                     '/vol/actrec/PAMAP/PAMAP2_Dataset/Protocol/subject103.dat', #2
                     '/vol/actrec/PAMAP/PAMAP2_Dataset/Protocol/subject104.dat', #3
                     '/vol/actrec/PAMAP/PAMAP2_Dataset/Protocol/subject105.dat', #4
                     '/vol/actrec/PAMAP/PAMAP2_Dataset/Protocol/subject106.dat', #5
                     '/vol/actrec/PAMAP/PAMAP2_Dataset/Protocol/subject107.dat', #6
                     '/vol/actrec/PAMAP/PAMAP2_Dataset/Protocol/subject108.dat', #7
                     '/vol/actrec/PAMAP/PAMAP2_Dataset/Protocol/subject109.dat', #8
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

def select_columns_opp(raw_data):
        """Selection of the columns employed in the Pamap2 dataset

        :param data: numpy integer matrix
            Sensor data (all features)
        :return: numpy integer matrix
            Selection of features
        """

        #                     included-excluded
        features_delete = np.arange(14, 18)
        features_delete = np.concatenate([features_delete, np.arange(31, 35)])
        features_delete = np.concatenate([features_delete, np.arange(48, 52)])

        return np.delete(raw_data, features_delete, 1)
    
def complete_HR(raw_data):

        pos_NaN = np.isnan(raw_data)
        idx_NaN = np.where(pos_NaN == False)[0]
        data_no_NaN = raw_data * 0
        for idx in range(idx_NaN.shape[0] - 1):
            data_no_NaN[idx_NaN[idx]: idx_NaN[idx + 1]] = raw_data[idx_NaN[idx]]

        data_no_NaN[idx_NaN[-1]:] = raw_data[idx_NaN[-1]]

        return data_no_NaN

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

def normalize(raw_data, max_list, min_list):
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
        for i in np.arange(raw_data.shape[1]):
            raw_data[:, i] = (raw_data[:, i] - min_list[i]) / diffs[i]
        #     Checking the boundaries
        raw_data[raw_data > 1] = 0.99
        raw_data[raw_data < 0] = 0.00
        return raw_data

def divide_x_y(raw_data):
        """Segments each sample into features and label

        :param data: numpy integer matrix
            Sensor data
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer matrix, numpy integer array
            Features encapsulated into a matrix and labels as an array
        """
        data_t = raw_data[:, 0]
        data_y = raw_data[:, 1]
        data_x = raw_data[:, 2:]

        return data_t, data_x, data_y

def del_labels(data_t, data_x, data_y):

        idy = np.where(data_y == 0)[0]
        labels_delete = idy

        idy = np.where(data_y == 8)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        idy = np.where(data_y == 9)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        idy = np.where(data_y == 10)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        idy = np.where(data_y == 11)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        idy = np.where(data_y == 18)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        idy = np.where(data_y == 19)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        idy = np.where(data_y == 20)[0]
        labels_delete = np.concatenate([labels_delete, idy])

        return np.delete(data_t, labels_delete, 0), np.delete(data_x, labels_delete, 0), np.delete(data_y,
                                                                                                   labels_delete, 0)
    
def adjust_idx_labels(data_y):
        """Transforms original labels into the range [0, nb_labels-1]

        :param data_y: numpy integer array
            Sensor labels
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer array
            Modified sensor labels
        """

        data_y[data_y == 24] = 0
        data_y[data_y == 12] = 8
        data_y[data_y == 13] = 9
        data_y[data_y == 16] = 10
        data_y[data_y == 17] = 11

        return data_y

def process_dataset_file(raw_data):
        """Function defined as a pipeline to process individual OPPORTUNITY files

        :param data: numpy integer matrix
            Matrix containing data samples (rows) for every sensor channel (column)
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer matrix, numy integer array
            Processed sensor data, segmented into features (x) and labels (y)
        """

        # Colums are segmentd into features and labels
        data_t, data_x, data_y = divide_x_y(raw_data)
        data_t, data_x, data_y = del_labels(data_t, data_x, data_y)

        data_y = adjust_idx_labels(data_y)
        data_y = data_y.astype(int)

        # Select correct columns
        data_x = select_columns_opp(data_x)

        if data_x.shape[0] != 0:
            HR_no_NaN = complete_HR(data_x[:, 0])
            data_x[:, 0] = HR_no_NaN

            data_x[np.isnan(data_x)] = 0
            # All sensor channels are normalized
            data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

        #data_t, data_x, data_y = self.downsampling(data_t, data_x, data_y)

        return data_x, data_y


def generate_data(target_filename):
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    """
    :param target_filename: string
        Processed file
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    :param datatype: string ['train', 'val', 'test']
        for choosing the sub categories for saving files
    """
    X = np.empty((0, NB_SENSOR_CHANNELS))
    Y = np.empty((0))
    lid= np.empty((0))
    
    X_train = np.empty((0, NB_SENSOR_CHANNELS))
    act_train = np.empty((0))
    id_train = np.empty((0))
    
    X_val = np.empty((0, NB_SENSOR_CHANNELS))
    act_val = np.empty((0))
    id_val = np.empty((0))
    
    X_test = np.empty((0, NB_SENSOR_CHANNELS))
    act_test = np.empty((0))
    id_test = np.empty((0))
    
    print('Processing dataset files ...')
    counter=0
    for idx_f in PAMAP2_DATA_FILES:
            try:
                print('Loading file...{0}'.format(idx_f))
                raw_data = np.loadtxt(idx_f)
                print(idx_f)
                x, y = process_dataset_file(raw_data)
                print("print datashape")
                print(x.shape)
                print(y.shape)
                
                shape=y.shape[0]
                train_no=round(0.64*shape)
                val_no=round(0.18*shape)
                tv= train_no+val_no
                
                x_train=x[0:train_no,:]
                x_val= x[train_no:tv,:]
                x_test= x[tv:shape,:]
                
                print(x_train.shape)
                
                a_train=y[0:train_no]
                a_val=y[train_no:tv]
                a_test=y[tv:shape]
                
                i_train=np.full(a_train.shape,counter)
                i_val=np.full(a_val.shape,counter)
                i_test=np.full(a_test.shape,counter)
                
                X = np.vstack((X, x))
                Y = np.concatenate([Y, y])
                lid = np.concatenate([lid, np.full(y.shape,counter)])
                
                X_train= np.vstack((X_train, x_train))
                act_train= np.concatenate([act_train, a_train])
                id_train= np.concatenate([id_train, i_train])
                
                X_val= np.vstack((X_val, x_val))
                act_val= np.concatenate([act_val, a_val])
                id_val= np.concatenate([id_val, i_val])
                
                X_test= np.vstack((X_test, x_test))
                act_test= np.concatenate([act_test, a_test])
                id_test= np.concatenate([id_test, i_test])
                
                counter+=1
            except KeyError:
                logging.error('ERROR: Did not find {0} in zip file'.format(PAMAP2_DATA_FILES[idx_f]))
     
    try:    
        data_train, act_train, act_all_train, labelid_train, labelid_all_train = opp_sliding_window(X_train, act_train, id_train, label_pos_end = False)
        data_val, act_val, act_all_val, labelid_val, labelid_all_val = opp_sliding_window(X_val, act_val, id_val, label_pos_end = False)
        data_test, act_test, act_all_test, labelid_test, labelid_all_test = opp_sliding_window(X_test, act_test, id_test, label_pos_end = False)
    except:
        print("error in sliding window")
        
    try:
        
        print("window extraction begining")
        
        print("training data save")
        print("target file name")
        print(data_dir_train)
        counter_seq = 0
        for f in range(data_train.shape[0]):
            try:
                sys.stdout.write('\r' + 'Creating sequence file '
                                 'number {} with id {}'.format(f, counter_seq))
                sys.stdout.flush()

                # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                seq = np.reshape(data_train[f], newshape = (1, data_train.shape[1], data_train.shape[2]))
                seq = np.require(seq, dtype=np.float)
                # Storing the sequences
                #obj = {"data": seq, "label": labelid}
                print("input values are")
                print(seq.shape)
                print(act_train[f].shape)
                print(act_train[f])
                print(labelid_train[f])
                obj = {"data": seq, "act_label": act_train[f], "act_labels_all": act_all_train[f], "label": labelid_train[f]}
                
                f = open(os.path.join(data_dir_train, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()

                counter_seq += 1
            except:
                raise('\nError adding the seq')
                
        print("val data save")
        print("target file name")
        print(data_dir_val)
        counter_seq = 0
        for f in range(data_val.shape[0]):
            try:
                sys.stdout.write('\r' + 'Creating sequence file '
                                 'number {} with id {}'.format(f, counter_seq))
                sys.stdout.flush()

                # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                seq = np.reshape(data_val[f], newshape = (1, data_val.shape[1], data_val.shape[2]))
                seq = np.require(seq, dtype=np.float)
                # Storing the sequences
                #obj = {"data": seq, "label": labelid}
                print("input values are")
                print(seq.shape)
                print(act_val[f].shape)
                print(act_val[f])
                print(labelid_val[f])
                obj = {"data": seq, "act_label": act_val[f], "act_labels_all": act_all_val[f], "label": labelid_val[f]}
                
                f = open(os.path.join(data_dir_val, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()

                counter_seq += 1
            except:
                raise('\nError adding the seq')
                
        print("test data save")
        print("target file name")
        print(data_dir_test)
        counter_seq = 0
        for f in range(data_test.shape[0]):
            try:
                sys.stdout.write('\r' + 'Creating sequence file '
                                 'number {} with id {}'.format(f, counter_seq))
                sys.stdout.flush()

                # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                seq = np.reshape(data_test[f], newshape = (1, data_test.shape[1], data_test.shape[2]))
                seq = np.require(seq, dtype=np.float)
                # Storing the sequences
                #obj = {"data": seq, "label": labelid}
                print("input values are")
                print(seq.shape)
                print(act_test[f].shape)
                print(act_test[f])
                print(labelid_test[f])
                obj = {"data": seq, "act_label": act_test[f], "act_labels_all": act_all_test[f], "label": labelid_test[f]}
                
                f = open(os.path.join(data_dir_test, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
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
    return

def generate_CSV(csv_dir, data_dir):
    '''
    Generate CSV file with path to all (Training) of the segmented sequences
    This is done for the DATALoader for Torch, using a CSV file with all the paths from the extracted
    sequences.

    @param csv_dir: Path to the dataset
    @param data_dir: Path of the training data
    '''
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for n in range(len(filenames)):
            f.append(data_dir + 'seq_{0:06}.pkl'.format(n))

    #np.savetxt(csv_dir + type_file, f, delimiter="\n", fmt='%s')
    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')
    
    return

def generate_CSV_final(csv_dir, data_dir1, data_dir2):
    '''
    Generate CSV file with path to all (Training and Validation) of the segmented sequences
    This is done for the DATALoader for Torch, using a CSV file with all the paths from the extracted
    sequences.

    @param csv_dir: Path to the dataset
    @param data_dir1: Path of the training data
    @param data_dir2: Path of the validation data
    '''
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir1):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:06}.pkl'.format(n))

    for dirpath, dirnames, filenames in os.walk(data_dir2):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:06}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')

    return
    
if __name__ == '__main__':
    
    base_directory = '/data/nnair/pamap/input2/'
    
    generate_data(base_directory)
    
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    
    generate_CSV(base_directory + "train.csv", data_dir_train)
    generate_CSV(base_directory + "val.csv", data_dir_val)
    generate_CSV(base_directory + "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)
    
    print("Done")