# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:18:36 2021

@author: nilah ravi nair
"""

import os
import sys
import numpy as np

import csv_reader
from sliding_window import sliding_window
import pickle

FOLDER_PATH = "/vol/actrec/DFG_Project/2019/LARa_dataset/MoCap/recordings_2019/14_Annotated_Dataset_renamed/"

NUM_CLASSES=7

headers = ["sample", "label", "head_RX", "head_RY", "head_RZ", "head_TX", "head_TY", "head_TZ", "head_end_RX",
           "head_end_RY", "head_end_RZ", "head_end_TX", "head_end_TY", "head_end_TZ", "L_collar_RX", "L_collar_RY",
           "L_collar_RZ", "L_collar_TX", "L_collar_TY", "L_collar_TZ", "L_elbow_RX", "L_elbow_RY", "L_elbow_RZ",
           "L_elbow_TX", "L_elbow_TY", "L_elbow_TZ", "L_femur_RX", "L_femur_RY", "L_femur_RZ", "L_femur_TX",
           "L_femur_TY", "L_femur_TZ", "L_foot_RX", "L_foot_RY", "L_foot_RZ", "L_foot_TX", "L_foot_TY", "L_foot_TZ",
           "L_humerus_RX", "L_humerus_RY", "L_humerus_RZ", "L_humerus_TX", "L_humerus_TY", "L_humerus_TZ", "L_tibia_RX",
           "L_tibia_RY", "L_tibia_RZ", "L_tibia_TX", "L_tibia_TY", "L_tibia_TZ", "L_toe_RX", "L_toe_RY", "L_toe_RZ",
           "L_toe_TX", "L_toe_TY", "L_toe_TZ", "L_wrist_RX", "L_wrist_RY", "L_wrist_RZ", "L_wrist_TX", "L_wrist_TY",
           "L_wrist_TZ", "L_wrist_end_RX", "L_wrist_end_RY", "L_wrist_end_RZ", "L_wrist_end_TX", "L_wrist_end_TY",
           "L_wrist_end_TZ", "R_collar_RX", "R_collar_RY", "R_collar_RZ", "R_collar_TX", "R_collar_TY", "R_collar_TZ",
           "R_elbow_RX", "R_elbow_RY", "R_elbow_RZ", "R_elbow_TX", "R_elbow_TY", "R_elbow_TZ", "R_femur_RX",
           "R_femur_RY", "R_femur_RZ", "R_femur_TX", "R_femur_TY", "R_femur_TZ", "R_foot_RX", "R_foot_RY", "R_foot_RZ",
           "R_foot_TX", "R_foot_TY", "R_foot_TZ", "R_humerus_RX", "R_humerus_RY", "R_humerus_RZ", "R_humerus_TX",
           "R_humerus_TY", "R_humerus_TZ", "R_tibia_RX", "R_tibia_RY", "R_tibia_RZ", "R_tibia_TX", "R_tibia_TY",
           "R_tibia_TZ", "R_toe_RX", "R_toe_RY", "R_toe_RZ", "R_toe_TX", "R_toe_TY", "R_toe_TZ", "R_wrist_RX",
           "R_wrist_RY", "R_wrist_RZ", "R_wrist_TX", "R_wrist_TY", "R_wrist_TZ", "R_wrist_end_RX", "R_wrist_end_RY",
           "R_wrist_end_RZ", "R_wrist_end_TX", "R_wrist_end_TY", "R_wrist_end_TZ", "root_RX", "root_RY", "root_RZ",
           "root_TX", "root_TY", "root_TZ"]

SCENARIO = {'R01': 'L01', 'R02': 'L01', 'R03': 'L02', 'R04': 'L02', 'R05': 'L02', 'R06': 'L02', 'R07': 'L02',
            'R08': 'L02', 'R09': 'L02', 'R10': 'L02', 'R11': 'L02', 'R12': 'L02', 'R13': 'L02', 'R14': 'L02',
            'R15': 'L02', 'R16': 'L02', 'R17': 'L03', 'R18': 'L03', 'R19': 'L03', 'R20': 'L03', 'R21': 'L03',
            'R22': 'L03', 'R23': 'L03', 'R24': 'L03', 'R25': 'L03', 'R26': 'L03', 'R27': 'L03', 'R28': 'L03',
            'R29': 'L03', 'R30': 'L03'}

annotator = {"S01": "A17", "S02": "A03", "S03": "A08", "S04": "A06", "S05": "A12", "S06": "A13",
             "S07": "A05", "S08": "A17", "S09": "A03", "S10": "A18", "S11": "A08", "S12": "A11",
             "S13": "A08", "S14": "A06", "S15": "A05", "S16": "A05"}

repetition = ["N01", "N02"]

annotator_S01 = ["A17", "A12"]

labels_persons = {"S07": 7, "S08": 8, "S09": 9, "S10": 10, "S11": 11, "S12": 12, "S13": 13, "S14": 14}

NORM_MAX_THRESHOLDS = [392.85,    345.05,    311.295,    460.544,   465.25,    474.5,     392.85,
                       345.05,    311.295,   574.258,   575.08,    589.5,     395.81,    503.798,
                       405.9174,  322.9,     331.81,    338.4,     551.829,   598.326,   490.63,
                       667.5,     673.4,     768.6,     560.07,    324.22,    379.405,   193.69,
                       203.65,    159.297,   474.144,   402.57,    466.863,   828.46,    908.81,
                       99.14,    482.53,    381.34,    386.894,   478.4503,  471.1,     506.8,
                       420.04,    331.56,    406.694,   504.6,     567.22,    269.432,   474.144,
                       402.57,    466.863,   796.426,   863.86,    254.2,     588.38,    464.34,
                       684.77,    804.3,     816.4,     997.4,     588.38,    464.34,    684.77,
                       889.5,     910.6,    1079.7,     392.0247,  448.56,    673.49,    322.9,
                       331.81,    338.4,     528.83,    475.37,    473.09,    679.69,    735.2,
                       767.5,     377.568,   357.569,   350.501,   198.86,    197.66,    114.931,
                       527.08,    412.28,    638.503,   691.08,    666.66,    300.48,    532.11,
                       426.02,    423.84,    467.55,    497.1,     511.9,     424.76,    348.38,
                       396.192,   543.694,   525.3,     440.25,    527.08,    412.28,    638.503,
                       729.995,   612.41,    300.33,    535.94,    516.121,   625.628,   836.13,
                       920.7,     996.8,     535.94,    516.121,   625.628,   916.15,   1009.5,
                       1095.6,    443.305,   301.328,   272.984,   138.75,    151.84,    111.35]

NORM_MIN_THRESHOLDS = [-382.62, -363.81, -315.691, -472.2, -471.4, -152.398,
                       -382.62, -363.81, -315.691, -586.3, -581.46, -213.082,
                       -400.4931, -468.4, -409.871, -336.8, -336.2, -104.739,
                       -404.083, -506.99, -490.27, -643.29, -709.84, -519.774,
                       -463.02, -315.637, -405.5037, -200.59, -196.846, -203.74,
                       -377.15, -423.992, -337.331, -817.74, -739.91, -1089.284,
                       -310.29, -424.74, -383.529, -465.34, -481.5, -218.357,
                       -442.215, -348.157, -295.41, -541.82, -494.74, -644.24,
                       -377.15, -423.992, -337.331, -766.42, -619.98, -1181.528,
                       -521.9, -581.145, -550.187, -860.24, -882.35, -645.613,
                       -521.9, -581.145, -550.187, -936.12, -982.14, -719.986,
                       -606.395, -471.892, -484.5629, -336.8, -336.2, -104.739,
                       -406.6129, -502.94, -481.81, -669.58, -703.12, -508.703,
                       -490.22, -322.88, -322.929, -203.25, -203.721, -201.102,
                       -420.154, -466.13, -450.62, -779.69, -824.456, -1081.284,
                       -341.5005, -396.88, -450.036, -486.2, -486.1, -222.305,
                       -444.08, -353.589, -380.33, -516.3, -503.152, -640.27,
                       -420.154, -466.13, -450.62, -774.03, -798.599, -1178.882,
                       -417.297, -495.1, -565.544, -906.02, -901.77, -731.921,
                       -417.297, -495.1, -565.544, -990.83, -991.36, -803.9,
                       -351.1281, -290.558, -269.311, -159.9403, -153.482, -162.718]

def opp_sliding_window(data_x, data_y, ws, ss, label_pos_end=True):
    print("check1")
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y_labels=np.full(data_x.shape[0],data_y)
    print("check2")
    print(data_x.shape)
    print(data_y_labels.shape)
    return data_x.astype(np.float32), data_y_labels.astype(np.uint8)
    

def normalize(data):
    """
    Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    try:
        max_list, min_list = np.array(NORM_MAX_THRESHOLDS), np.array(NORM_MIN_THRESHOLDS)
        diffs = max_list - min_list
        for i in np.arange(data.shape[1]):
            data[:, i] = (data[:, i]-min_list[i])/diffs[i]
        #     Checking the boundaries
        data[data > 1] = 0.99
        data[data < 0] = 0.00
    except:
        raise("Error in normalization")
        
    return data

def divide_x_y(data):
    """Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """
    data_t = data[:, 0]
    data_y = data[:, 1]
    data_x = data[:, 2:]
    
    return data_t, data_x, data_y

def select_columns_opp(data):
    """
    Selection of the columns employed in the MoCAP
    excluding the measurements from lower back,
    as this became the center of the human body,
    and the rest of joints are normalized
    with respect to this one

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: numpy integer matrix
        Selection of features
    """

    #included-excluded
    features_delete = np.arange(68, 74)
    
    return np.delete(data, features_delete, 1)

def generate_data(ids, sliding_window_length, sliding_window_step, data_dir=None, identity_bool=False, usage_modus='train'):
    #type1-avoiding person 12
    persons = ["S07", "S08", "S09", "S10", "S11", "S13", "S14"]
    ID = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S11": 4, "S13": 5, "S14": 6}
    train_ids = ["R03", "R07", "R08", "R10","R11"]
    val_ids = ["R12"]
    test_ids = ["R15"]
    
    #type2- avoiding person 11
    '''
    persons = ["S07", "S08", "S09", "S10", "S12", "S13", "S14"]
    ID = {"S07": 7, "S08": 8, "S09": 9, "S10": 10, "S11": 11, "S13": 13, "S14": 14}    
    train_ids = ["R11", "R12", "R13", "R15", "R18"]
    val_ids = ["R19", "R21"]
    test_ids = ["R22", "R23"]
    '''
    
    #type3- Avoiding person 11 and 12
    '''
    persons = ["S07", "S08", "S09", "S10", "S13", "S14"]
     ID = {"S07": 7, "S08": 8, "S09": 9, "S10": 10, "S11": 11, "S13": 13, "S14": 14}
     train_ids = ["R03", "R07", "R08", "R10" "R11", "R12", "R15", "R18"]
    val_ids = ["R19", "R21"]
    test_ids = ["R22", "R23"]
    '''
    
    #type4-Avoiding persons 11,12,10
    '''
    persons = ["S07", "S08", "S09", "S13", "S14"]
     ID = {"S07": 7, "S08": 8, "S09": 9, "S10": 10, "S11": 11, "S13": 13, "S14": 14}
     train_ids = ["R03", "R07", "R08", "R10" "R11", "R12", "R15", "R18", "R19", "R21", "R22"]
    val_ids = ["R23","R25", "R26"]
    test_ids = ["R27", "R28", "R29"]
    '''
    counter_seq = 0
    
    for P in persons:
        if usage_modus == 'train':
           recordings = train_ids
        elif usage_modus == 'val':
            recordings = val_ids
        elif usage_modus == 'test':
            recordings = test_ids
        print("\nModus {} \n{}".format(usage_modus, recordings))
        for R in recordings:
            S = SCENARIO[R]
            for N in repetition:
                    annotator_file = annotator[P]
                    if P == 'S07' and SCENARIO[R] == 'L01':
                        annotator_file = "A03"
                    if P == 'S14' and SCENARIO[R] == 'L03':
                        annotator_file = "A19"
                    if P == 'S11' and SCENARIO[R] == 'L01':
                        annotator_file = "A03"
                    if P == 'S11' and R in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                        annotator_file = "A02"
                    if P == 'S13' and R in ['R28']:
                        annotator_file = "A01"
                    if P == 'S13' and R in ['R29', 'R30']:
                        annotator_file = "A11"
                    if P == 'S09' and R in ['R28', 'R29']:
                        annotator_file = "A01"
                    if P == 'S09' and R in ['R21', 'R22', 'R23', 'R24', 'R25']:
                        annotator_file = "A11"
                        
                    file_name_norm = "{}/{}_{}_{}_{}_{}_norm_data.csv".format(P, S, P, R, annotator_file, N)
                    try:
                        data = csv_reader.reader_data(FOLDER_PATH + file_name_norm)
                        print("\nFiles loaded in modus {}\n{}".format(usage_modus, file_name_norm))
                        data = select_columns_opp(data)
                        print("\nFiles loaded")
                    except:
                        print("\n1 In loading data,  in file {}".format(FOLDER_PATH + file_name_norm))
                        continue
                    
                    label=ID[P]
                    
                    data_t, data_x, data_y = divide_x_y(data)
                    del data_t
                    #    print("\n In generating data, Error getting the data {}".format(FOLDER_PATH + file_name_norm))
                    #    continue
                    try:
                        # checking if annotations are consistent
                        #data_x = normalize(data_x)
                        print("data shape")
                        print(data_x.shape)
                        if data_x.shape[0] == data_x.shape[0]:
                            print("Starting sliding window")
                            X, y= opp_sliding_window(data_x, label.astype(int),
                                                             sliding_window_length,
                                                             sliding_window_step, label_pos_end = False)
                            print("Windows are extracted")
                            
                            for f in range(X.shape[0]):
                                try:

                                    sys.stdout.write('\r' + 'Creating sequence file '
                                                            'number {} with id {}'.format(f, counter_seq))
                                    sys.stdout.flush()

                                    # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                                    seq = np.reshape(X[f], newshape = (1, X.shape[1], X.shape[2]))
                                    seq = np.require(seq, dtype=np.float)

                                    # Storing the sequences
                                    obj = {"data": seq, "label": y[f]}
                                    f = open(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                                    f.close()

                                    counter_seq += 1
                                except:
                                    raise('\nError adding the seq')

                            print("\nCorrect data extraction from {}".format(FOLDER_PATH + file_name_norm))

                            del data
                            del data_x
                            del data_y
                            del X
                            del labels
                            del class_labels

                        else:
                            print("\nNot consisting annotation in  {}".format(file_name_norm))
                            continue

                    except:
                        print("\n In generating data, No file {}".format(FOLDER_PATH + file_name_norm))
            
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
                            
                            
def create_dataset():
    #type1-avoiding person 12
    train_ids = ["R03", "R07", "R08", "R10"]
    val_ids = ["R12"]
    test_ids = ["R15"]
    
    #type2- avoiding person 11
    '''
    train_ids = ["R11", "R12", "R13", "R15", "R18"]
    val_ids = ["R19", "R21"]
    test_ids = ["R22", "R23"]
    '''
    
    #type3- Avoiding person 11 and 12
    '''
    train_ids = ["R03", "R07", "R08", "R10" "R11", "R12", "R15", "R18"]
    val_ids = ["R19", "R21"]
    test_ids = ["R22", "R23"]
    '''
    
    #type4-Avoiding persons 11,12,10
    '''
    train_ids = ["R03", "R07", "R08", "R10" "R11", "R12", "R15", "R18", "R19", "R21", "R22"]
    val_ids = ["R23","R25", "R26"]
    test_ids = ["R27", "R28", "R29"]
    '''
    
    base_directory = '/data/nnair/output/type1/mocap/'
    sliding_window_length = 100
    sliding_window_step = 25
    
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    
    generate_data(train_ids, sliding_window_length=sliding_window_length,
                  sliding_window_step=sliding_window_step, data_dir=data_dir_train, usage_modus='train')
    generate_data(val_ids, sliding_window_length=sliding_window_length,
                  sliding_window_step=sliding_window_step, data_dir=data_dir_val, usage_modus='val')
    generate_data(test_ids, sliding_window_length=sliding_window_length,
                  sliding_window_step=sliding_window_step, data_dir=data_dir_test, usage_modus='test')
    
    generate_CSV(base_directory + "train.csv", data_dir_train)
    generate_CSV(base_directory + "val.csv", data_dir_val)
    generate_CSV(base_directory + "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)
    
    return


#######################

if __name__ == '__main__':
    
    create_dataset()

    print("Done")