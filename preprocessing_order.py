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
import theano

from io import BytesIO
from pandas import Series

import numpy as np
import sys
#import torch.utils.data as data
import logging
from sliding_window_dat import sliding_window
#from resampling import Resampling



# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 27
NUM_ACT_CLASSES= 11
NUM_CLASSES =5

ws = 100
ss = 12

labels_dict = {0: "NULL", 1: "UNKNOWN", 2: "FLIP", 3: "WALK",
                       4: "SEARCH", 5: "PICK", 6: "SCAN", 7: "INFO",
                       8: "COUNT", 9: "CARRY", 10: "ACK"}

location= '/vol/actrec/icpram-data/numpy_arrays/'
dictz = {"_DO": {1: "004", 2: "011", 3: "017"}, "_NP": {1: "004", 2: "014", 3: "015"}}

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
    
    X_train = np.empty((0, 100, NB_SENSOR_CHANNELS))
    act_train = np.empty((0))
    id_train = np.empty((0))
    
    X_val = np.empty((0, 100, NB_SENSOR_CHANNELS))
    act_val = np.empty((0))
    id_val = np.empty((0))
    
    X_test = np.empty((0, 100, NB_SENSOR_CHANNELS))
    act_test = np.empty((0))
    id_test = np.empty((0))
    
    wr='_DO'
    totaldata = list((dictz[wr]).keys())
    print(totaldata)
    totaldata_list = ["/vol/actrec/icpram-data/numpy_arrays/" + "%s__%s_data_labels_every-frame_100.npz" %(
        wr, dictz[wr][ totaldata[i]]) for i in [0, 1, 2]]
    print("totaldatalist")
    print(totaldata_list)
    wr='_NP'
    adddata=list((dictz[wr]).keys())
    adddata_list= ["/vol/actrec/icpram-data/numpy_arrays/"  + "%s__%s_data_labels_every-frame_100.npz" % (wr, dictz[wr][adddata[i]]) for i in [0, 1, 2]]
    totaldata_list=totaldata_list+adddata_list
    print(totaldata_list)
    
    total_data = []
    total_labels = []
    total_id=[]
     
    print('Processing dataset files ...')
    counter=0
    for path in totaldata_list:
        print(path)
        tmp = np.load(path)
        data= tmp["arr_0"].copy()
        act_labels= tmp["arr_1"].copy()
        
        tmp.close()
        
       
        labels=[]
        #person_id=[]
        for i in range(len(act_labels)):
            
            label_arg = act_labels[i].flatten()
            label_arg = label_arg.astype(int)
            label_arg = label_arg[int(label_arg.shape[0]/2)]
            labels.append(label_arg)
            #person_id.append(counter)
        
        '''
        #calculate number of labels
        l=set([])
        l=l.union(set(labels))
        print("union l")
        print(l)
        # Remove NULL class label -> should be ignored
        l = sorted(l)
        if l[0] == 0:
            l = l[1:]
           
        print("sorted l")
        print(l)
        #
        # Create a class dictionary and save it
        # It is a mapping from the original labels
        # to the new labels, due that the all the
        # labels dont exist in the warehouses
        #
        #
        class_dict = {}
        for i,lab in enumerate(l):
            class_dict[lab]=i
        
        print("class_dict")
        print(class_dict)
        '''
        
        labels = np.array(labels)
        print("data shape")
        print(data.shape)
        print("labels shape")
        print(labels.shape)
        '''
        c = 0
        nonulldata = []
        nonulllabel = []
        for idx in range(labels.shape[0]):
            item = np.copy(data[idx])
            label = labels[idx]
            print(label)
            
            if label == 0:
                continue
            nonulldata.append(item)
            nonulllabel.append(int(class_dict[label]))
            
            c += 1
        '''
        #i_0=[]
        i_1=[]
        i_2=[]
        i_3=[]
        i_4=[]
        i_5=[]
        i_6=[]
        i_7=[]
        i_8=[]
        i_9=[]
        i_10=[]
        #i_11=[]
                
        yl_train=[]
        yl_val=[]
        yl_test=[]
                
        ##########put index of relating to each activity into respective group
        for i in range(labels.shape[0]):
            '''        
            if labels[i] == 0:
                i_0 = i_0 + [i]
            '''
            if labels[i] == 1:
                i_1 = i_1 + [i]
            elif labels[i] == 2:
                i_2 = i_2 + [i]
            elif labels[i] == 3:
                i_3 = i_3 + [i]
            elif labels[i] == 4:
                i_4 = i_4 + [i]
            elif labels[i] == 5:
                i_5 = i_5 + [i]
            elif labels[i] == 6:
                i_6 = i_6 + [i]
            elif labels[i] == 7:
                i_7 = i_7 + [i]
            elif labels[i] == 8:
                i_8 = i_8 + [i]
            elif labels[i] == 9:
                i_9 = i_9 + [i]
            elif labels[i] == 10:
                i_10 = i_10 + [i]
            '''
            elif labels[i] == 11:
                i_11 = i_11 + [i]
            '''
            ################split the index list and concatenate
                
        print("shape of the label index")
                
        #print(len(i_0))
        print(len(i_1))
        print(len(i_2))
        print(len(i_3))
        print(len(i_4))
        print(len(i_5))
        print(len(i_6))
        print(len(i_7))
        print(len(i_8))
        print(len(i_9))
        print(len(i_10))
        #print(len(i_11))
        
        '''       
        shape = len(i_0)
        if shape != 0:
            train_no=round(0.64*shape)
            val_no=round(0.18*shape)
            tv= train_no+val_no
            yl_train= np.concatenate([yl_train, i_0[0:train_no]])
            yl_val=  np.concatenate([yl_val, i_0[train_no:tv]])
            yl_test= np.concatenate([yl_test, i_0[tv:shape]])
        '''        
        shape = len(i_1)
        if shape != 0:
            train_no=round(0.64*shape)
            val_no=round(0.18*shape)
            tv= train_no+val_no
            yl_train= np.concatenate([yl_train, i_1[0:train_no]])
            yl_val= np.concatenate ([yl_val, i_1[train_no:tv]])
            yl_test= np.concatenate([yl_test, i_1[tv:shape]])
                
        shape = len(i_2)
        if shape !=0:
            train_no=round(0.64*shape)
            val_no=round(0.18*shape)
            tv= train_no+val_no
            yl_train= np.concatenate([yl_train, i_2[0:train_no]])
            yl_val= np.concatenate([yl_val, i_2[train_no:tv]])
            yl_test= np.concatenate([yl_test, i_2[tv:shape]])
                
        shape = len(i_3)
        if shape !=0:
            train_no=round(0.64*shape)
            val_no=round(0.18*shape)
            tv= train_no+val_no
            yl_train= np.concatenate([yl_train, i_3[0:train_no]])
            yl_val= np.concatenate([yl_val, i_3[train_no:tv]])
            yl_test= np.concatenate([yl_test, i_3[tv:shape]])
                
        shape =len(i_4)
        if shape !=0:
            train_no=round(0.64*shape)
            val_no=round(0.18*shape)
            tv= train_no+val_no
            yl_train= np.concatenate([yl_train, i_4[0:train_no]])
            yl_val= np.concatenate ([yl_val, i_4[train_no:tv]])
            yl_test= np.concatenate([yl_test, i_4[tv:shape]])
                
        shape = len(i_5)
        if shape !=0:
            train_no=round(0.64*shape)
            val_no=round(0.18*shape)
            tv= train_no+val_no
            yl_train= np.concatenate([yl_train, i_5[0:train_no]])
            yl_val= np.concatenate([yl_val, i_5[train_no:tv]])
            yl_test= np.concatenate([yl_test, i_5[tv:shape]])
                
        shape = len(i_6)
        if shape != 0:
            train_no=round(0.64*shape)
            val_no=round(0.18*shape)
            tv= train_no+val_no
            yl_train= np.concatenate([yl_train, i_6[0:train_no]])
            yl_val= np.concatenate ([yl_val, i_6[train_no:tv]])
            yl_test= np.concatenate([yl_test, i_6[tv:shape]])
                
        shape = len(i_7)
        if shape != 0:
            train_no=round(0.64*shape)
            val_no=round(0.18*shape)
            tv= train_no+val_no
            yl_train= np.concatenate([yl_train, i_7[0:train_no]])
            yl_val= np.concatenate([yl_val, i_7[train_no:tv]])
            yl_test= np.concatenate([yl_test, i_7[tv:shape]])
            
        shape = len(i_8)
        if shape !=0:
            train_no=round(0.64*shape)
            val_no=round(0.18*shape)
            tv= train_no+val_no
            yl_train= np.concatenate([yl_train, i_8[0:train_no]])
            yl_val= np.concatenate([yl_val, i_8[train_no:tv]])
            yl_test= np.concatenate([yl_test, i_8[tv:shape]])
                
        shape = len(i_9)
        if shape !=0:
            train_no=round(0.64*shape)
            val_no=round(0.18*shape)
            tv= train_no+val_no
            yl_train= np.concatenate([yl_train, i_9[0:train_no]])
            yl_val= np.concatenate([yl_val, i_9[train_no:tv]])
            yl_test= np.concatenate([yl_test, i_9[tv:shape]])
                
        shape = len(i_10)
        if shape !=0:
            train_no=round(0.64*shape)
            val_no=round(0.18*shape)
            tv= train_no+val_no
            yl_train= np.concatenate([yl_train, i_10[0:train_no]])
            yl_val= np.concatenate([yl_val, i_10[train_no:tv]])
            yl_test= np.concatenate([yl_test, i_10[tv:shape]])
        '''    
        shape = len(i_11)
        if shape !=0:
            train_no=round(0.64*shape)
            val_no=round(0.18*shape)
            tv= train_no+val_no
            yl_train= np.concatenate([yl_train, i_11[0:train_no]])
            yl_val= np.concatenate([yl_val, i_11[train_no:tv]])
            yl_test= np.concatenate([yl_test, i_11[tv:shape]])
        '''        
        print("yl_train.shape")
        print(yl_train.shape)
        print("yl_val.shape")
        print(yl_val.shape)
        print("yl_test.shape")
        print(yl_test.shape)
    
        print(yl_train[1])
        print(len([data[int(yl_train[1])]]))
        print(data[int(yl_train[1]),:,:])
        
        if counter >= 3:
            if counter == 3:
                cter = 0
            else:
                cter = counter -1
        else:
            cter = counter
    
        for i in range(len(yl_train)):
            X_train = np.append(X_train, [data[int(yl_train[i]),:,:]], axis=0)
            act_train = np.append(act_train, [labels[int(yl_train[i])]])
            id_train = np.append(id_train, cter)
        
        print(X_train.shape)
        print("debug")
        print(act_train.shape)
        print(id_train.shape)
        print("id ")
        print(id_train)
    
        for i in range(len(yl_val)):
            X_val = np.append(X_val, [data[int(yl_val[i]),:,:]], axis=0)
            act_val = np.append(act_val, [labels[int(yl_val[i])]])
            id_val = np.append(id_val, cter)
                    
        for i in range(len(yl_test)):
            X_test = np.append(X_test, [data[int(yl_test[i]),:,:]], axis=0)
            act_test = np.append(act_test, [labels[int(yl_test[i])]])
            id_test = np.append(id_test, cter)
                
        print("shape of final products train")    
        print(X_train.shape)
        print(act_train.shape)
        print(id_train.shape) 
                
        print("shape of final products val")    
        print(X_val.shape)
        print(act_val.shape)
        print(id_val.shape)
                
        print("shape of final products test")    
        print(X_test.shape)
        print(act_test.shape)
        print(id_test.shape)
        
        '''
        #test_vals_fl = np.array(test_vals_fl)
        data = np.array(nonulldata)
        labels = np.array(nonulllabel)
        person_id = np.full(labels.shape, counter)
        
        
        
        print("shape of data, act_labels and id")
        print(data.shape)
        print(labels.shape)
        print(person_id.shape)
        
        shape=len(labels)
        train_no=round(0.64*shape)
        val_no=round(0.18*shape)
        tv= train_no+val_no
             
        x_train=data[0:train_no,:]
        x_val= data[train_no:tv,:]
        x_test= data[tv:shape,:]
        print("division into train val test")        
        print(x_train.shape)
        
        a_train=labels[0:train_no]
        a_val=labels[train_no:tv]
        a_test=labels[tv:shape]
                
        print(a_train.shape)
        
        i_train= person_id[0:train_no]
        i_val= person_id[train_no:tv]
        i_test= person_id[tv:shape]
        
        print(i_train)
        
        X_train= np.concatenate((X_train, x_train), axis=0)
        act_train= np.concatenate([act_train, a_train])
        id_train= np.concatenate([id_train, i_train])
        print("X_train.shape")
        print(X_train.shape)
                
        X_val= np.vstack((X_val, x_val))
        act_val= np.concatenate([act_val, a_val])
        id_val= np.concatenate([id_val, i_val])
        
        print("X_val.shape")
        print(X_val.shape)
                
        X_test= np.vstack((X_test, x_test))
        act_test= np.concatenate([act_test, a_test])
        id_test= np.concatenate([id_test, i_test])
        
        print("X_test.shape")
        print(X_test.shape)
        '''
        counter+=1
        
    # Make train arrays a numpy matrix

        ##############################
    # Normalizing the data to be in range [0,1] following the paper
    '''
    total = np.vstack((X_train, X_val, X_test))
    for ch in range(total.shape[2]):
        max_ch = np.max(total[:, :, ch])
        min_ch = np.min(total[:, :, ch])
        median_old_range = (max_ch + min_ch) / 2
        X_train[:, :, ch] = (X_train[:, :, ch] - median_old_range) / (max_ch - min_ch)  # + 0.5
        X_val[:, :, ch] = (X_val[:, :, ch] - median_old_range) / (max_ch - min_ch)
        X_test[:, :, ch] = (X_test[:, :, ch] - median_old_range) / (max_ch - min_ch)
        
    '''
    
    '''   
    for ch in range(X_val.shape[2]):
        max_ch = np.max(X_val[:, :, ch])
        min_ch = np.min(X_val[:, :, ch])
        median_old_range = (max_ch + min_ch) / 2
        X_val[:, :, ch] = (X_val[:, :, ch] - median_old_range) / (max_ch - min_ch)  # + 0.5
        
    for ch in range(X_test.shape[2]):
        max_ch = np.max(X_test[:, :, ch])
        min_ch = np.min(X_test[:, :, ch])
        median_old_range = (max_ch + min_ch) / 2
        X_test[:, :, ch] = (X_test[:, :, ch] - median_old_range) / (max_ch - min_ch)  # + 0.5
    '''
    '''
    try:    
        data_train, act_train, act_all_train, labelid_train, labelid_all_train = opp_sliding_window(X_train, act_train, id_train, label_pos_end = False)
        data_val, act_val, act_all_val, labelid_val, labelid_all_val = opp_sliding_window(X_val, act_val, id_val, label_pos_end = False)
        data_test, act_test, act_all_test, labelid_test, labelid_all_test = opp_sliding_window(X_test, act_test, id_test, label_pos_end = False)
    except:
        print("error in sliding window")
    '''    
    try:
        
        print("window extraction begining")
        
        print("training data save")
        print("target file name")
        print(data_dir_train)
        counter_seq = 0
        for f in range(X_train.shape[0]):
            try:
                sys.stdout.write('\r' + 'Creating sequence file '
                                 'number {} with id {}'.format(f, counter_seq))
                sys.stdout.flush()

                # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                seq = np.reshape(X_train[f], newshape = (1, X_train.shape[1], X_train.shape[2]))
                seq = np.require(seq, dtype=np.float)
                # Storing the sequences
                #obj = {"data": seq, "label": labelid}
                print("input values are")
                print(seq.shape)
                print(act_train[f])
                print(id_train[f])
                obj = {"data": seq, "act_label": act_train[f], "label": id_train[f]}
                
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
        for f in range(X_val.shape[0]):
            try:
                sys.stdout.write('\r' + 'Creating sequence file '
                                 'number {} with id {}'.format(f, counter_seq))
                sys.stdout.flush()

                # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                seq = np.reshape(X_val[f], newshape = (1, X_val.shape[1], X_val.shape[2]))
                seq = np.require(seq, dtype=np.float)
                # Storing the sequences
                #obj = {"data": seq, "label": labelid}
                print("input values are")
                print(seq.shape)
                print(act_val[f])
                print(id_val[f])
                obj = {"data": seq, "act_label": act_val[f], "label": id_val[f]}
                
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
        for f in range(X_test.shape[0]):
            try:
                sys.stdout.write('\r' + 'Creating sequence file '
                                 'number {} with id {}'.format(f, counter_seq))
                sys.stdout.flush()

                # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                seq = np.reshape(X_test[f], newshape = (1, X_test.shape[1], X_test.shape[2]))
                seq = np.require(seq, dtype=np.float)
                # Storing the sequences
                #obj = {"data": seq, "label": labelid}
                print("input values are")
                print(seq.shape)
                print(act_test[f])
                print(id_test[f])
                obj = {"data": seq, "act_label": act_test[f], "label": id_test[f]}
                
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
    
    #base_directory = '/data/nnair/all/order/'
    base_directory = '/data/nnair/all/nonormorder/'
    
    generate_data(base_directory)
    
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    
    generate_CSV(base_directory + "train.csv", data_dir_train)
    generate_CSV(base_directory + "val.csv", data_dir_val)
    generate_CSV(base_directory + "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)
    
    print("Done")