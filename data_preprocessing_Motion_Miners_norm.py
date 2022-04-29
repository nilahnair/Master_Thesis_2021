# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 18:00:30 2021

@author: nilah

Code idea by Fernando Moya: https://github.com/wilfer9008/Annotation_Tool_LARa/tree/master/From_Human_Pose_to_On_Body_Devices_for_Human_Activity_Recognition
"""

import numpy as np
import csv
import os
import sys
import datetime
import csv_reader
from sliding_window import sliding_window
import pickle

FOLDER_PATH = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/flw_recordings_annotated/"

NUM_CLASSES=8
SCENARIO = {'R01': 'L01', 'R02': 'L01', 'R03': 'L02', 'R04': 'L02', 'R05': 'L02', 'R06': 'L02', 'R07': 'L02',
            'R08': 'L02', 'R09': 'L02', 'R10': 'L02', 'R11': 'L02', 'R12': 'L02', 'R13': 'L02', 'R14': 'L02',
            'R15': 'L02', 'R16': 'L02', 'R17': 'L03', 'R18': 'L03', 'R19': 'L03', 'R20': 'L03', 'R21': 'L03',
            'R22': 'L03', 'R23': 'L03', 'R24': 'L03', 'R25': 'L03', 'R26': 'L03', 'R27': 'L03', 'R28': 'L03',
            'R29': 'L03', 'R30': 'L03'}
labels_persons = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S11": 4, "S12": 5, "S13": 6, "S14": 7}
NUM_CLASSES = 8

def opp_sliding_window(data_x, data_y, ws, ss, label_pos_end=True):
    '''
    Performs the sliding window approach on the data and the labels

    return three arrays.
    - data, an array where first dim is the windows
    - labels per window according to end, middle or mode
    - all labels per window

    @param data_x: ids for train
    @param data_y: ids for train
    @param ws: ids for train
    @param ss: ids for train
    @param label_pos_end: ids for train
    '''

    print("Sliding window: Creating windows {} with step {}".format(ws, ss))

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    count_l = 0
    idy = 0
    # Label from the end
    if label_pos_end:
        data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
    else:

        # Label from the middle
        if False:
            data_y_labels = np.asarray(
                [[i[i.shape[0] // 2]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
        else:

            # Label according to mode
            try:
                data_y_labels = []
                for sw in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1)):
                    labels = np.zeros((20)).astype(int)
                    count_l = np.bincount(sw[:, 0], minlength=NUM_CLASSES)
                    idy = np.argmax(count_l)
                    attrs = np.sum(sw[:, 1:], axis=0)
                    attrs[attrs > 0] = 1
                    labels[0] = idy
                    labels[1:] = attrs
                    data_y_labels.append(labels)
                data_y_labels = np.asarray(data_y_labels)


            except:
                print("Sliding window: error with the counting {}".format(count_l))
                print("Sliding window: error with the counting {}".format(idy))
                return np.Inf

            # All labels per window
            data_y_all = np.asarray([i[:] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])

    return data_x.astype(np.float32), data_y_labels.astype(np.uint8), data_y_all.astype(np.uint8)


def norm_motion_miners(data):
    """
    Normalizes all sensor channels
    Zero mean and unit variance

    @param data: numpy integer matrix
    @return data_norm: Normalized sensor data
    """
#change
    '''
    mean_values = np.array([-0.6018319,   0.234877,    0.2998928,   1.11102944,  0.17661719, -1.41729978,
                   0.03774093,  1.0202137,  -0.1362719,   1.78369919,  2.4127946,  -1.36437627,
                   -0.96302063, -0.0836716,   0.13035097,  0.08677377,  1.088766,    0.51141513,
                   -0.61147614, -0.22219321,  0.41094977, -1.45036893,  0.80677986, -0.1342488,
                   -0.02994514, -0.999678,   -0.22073192, -0.1808128,  -0.01197039,  0.82491874])
    mean_values = np.reshape(mean_values, [1, 30])

    std_values = np.array([1.17989719,   0.55680584,   0.65610454,  58.42857495,  74.36437559,
                  86.72291263,   1.01306,      0.62489802,   0.70924608,  86.47014857,
                  100.6318856,   61.02139095,   0.38256693,   0.21984504,   0.32184666,
                  42.84023413,  24.85339931,  18.02111335,   0.44021448,   0.51931148,
                  0.45731142,  78.58164965,  70.93038919,  76.34418105,   0.78003314,
                  0.32844988,   0.54919488,  26.68953896,  61.04472454,  62.9225945])
    '''
    #all 
    
    mean_values=np.array([-1972.21093,  1319.18494,  2086.44939,  3.47782711,
                          9.52718419, -15.4629411,  2839.43096, -2485.97015,
                          -2057.10843,  106.437329, -3742.25358, -879.644817,
                          -7.07292913,  7.37774481, -3.66716486, -184.263412,
                          1747.75134,  528.392669,  1963.13183,  1187.77773,
                          2204.39909,  4.49233848, -13.9301357,  19.7020469,
                          -2115.97999, -2036.88632, -2350.28944])
    mean_values = np.reshape(mean_values, [1, 27])
    
    std_values = np.array([1659.81581324, 1922.83004563, 1403.7565496,   803.60068613,  994.99489569,
                           901.08895607, 2186.96453082, 2738.03382675, 1912.30333819,  812.34836189,
                           594.45217882, 1313.82300176,  280.76318532,  580.97537463,  256.97295422,
                           838.19728655,  423.39773434, 1117.36749732, 1744.18255968, 1942.39160641,
                           1383.33653,    861.84780237, 1036.19127535,  923.98531812, 2604.48576624,
                           2892.60854077, 1799.94353505])

    std_values = np.reshape(std_values, [1, 27])

    mean_array = np.repeat(mean_values, data.shape[0], axis=0)
    std_array = np.repeat(std_values, data.shape[0], axis=0)

    max_values = mean_array + 2 * std_array
    min_values = mean_array - 2 * std_array

    data_norm = (data - min_values) / (max_values - min_values)

    data_norm[data_norm > 1] = 1
    data_norm[data_norm < 0] = 0

    return data_norm

############################
'''
def reader_data(path):
    print('Getting data from {}'.format(path))
    #counter = 0
    IMU = []
    time = []
    data = []
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            try:
                try:
                    if spamreader.line_num == 1:
                        # print('\n')
                        print(', '.join(row))
                    else:
                        if len(row) != 29:
                            idx_row = 2
                            IMU.append(row[idx_row])
                            idx_row += 1
                        else:
                            idx_row = 0
                        try:
                            time_d = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
                            idx_row += 1
                        except:
                            try:
                                time_d = datetime.datetime.strptime(row[idx_row.astype(int)], '%Y-%m-%d %H:%M:%S')
                                idx_row += 1
                            except:
                                print("strange time str {}".format(time_d))
                                continue
                        time.append(time_d)
                        data.append(list(map(float, row[idx_row:])))
                except:
                    print("Error in line {}".format(row))
            except KeyboardInterrupt:
                print('\nYou cancelled the operation.')

    if len(row) != 29:
        imu_data = {'IMU': IMU, 'time': time, 'data': data}
    else:
        imu_data = {'time': time, 'data': data}
    return imu_data
'''
def read_extracted_data(path, skiprows = 1):
    '''
    gets data from csv file
    data contains 3 columns, start, end and label

    returns a numpy array

    @param path: path to file
    '''
    print("check2")
    annotation_original = np.loadtxt(path, delimiter=',', skiprows=skiprows)
    return annotation_original


#####################

def generate_data(ids, sliding_window_length, sliding_window_step, data_dir=None, identity_bool=False, usage_modus='train'):
    
   #type1-avoiding person 12
    '''
    persons = ["S07", "S08", "S09", "S10", "S11", "S13", "S14"]
    ID = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S11": 4, "S13": 5, "S14": 6}
    train_ids = ["R03", "R07", "R08", "R10", "R11"]
    val_ids = ["R12"]
    test_ids = ["R15"]
    '''
    #type2- avoiding person 11
    '''
    persons = ["S07", "S08", "S09", "S10", "S12", "S13", "S14"]
    ID = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S12": 4, "S13": 5, "S14": 6}
    train_ids =["R11", "R12", "R15", "R18", "R19","R21"]
    val_ids = ["R22"]
    test_ids = ["R23"]
    '''
    
    #type3- Avoiding person 11 and 12
    '''
    persons = ["S07", "S08", "S09", "S10", "S13", "S14"]
    ID = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S13": 4, "S14": 5}
    train_ids = ["R03", "R07", "R08", "R10", "R11", "R12", "R15", "R18"]
    val_ids = ["R19", "R21"]
    test_ids = ["R22", "R23"]
    '''
    '''
    #type4-Avoiding persons 11,12,10
    
    persons = ["S07", "S08", "S09", "S13", "S14"]
    ID = {"S07": 0, "S08": 1, "S09": 2, "S13": 3, "S14": 4}
    train_ids = ["R03", "R07", "R08", "R10", "R11", "R12", "R15", "R18", "R19", "R21", "R22"]
    val_ids = ["R23","R25", "R26"]
    test_ids = ["R27", "R28", "R29"]
    '''
    
    labels_persons = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S11": 4, "S12": 5, "S13": 6, "S14": 7}
    
    #persons = [ "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]
    if usage_modus == 'train':
           persons = [ "S07", "S08", "S09", "S10", "S11", "S12", "S14"]
    elif usage_modus == 'val':
           persons = ["S07", "S08", "S09", "S10", "S11", "S12", "S14"]
    elif usage_modus == 'test':
           persons = ["S13"]
           
    ID = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S11": 4, "S12":5, "S13": 6, "S14": 7}
    train_ids = ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", 
                 "R13", "R14", "R16", "R17", "R18", "R19", "R20", "R21", "R22", "R23", 
                 "R24", "R25", "R26", "R27", "R28", "R29", "R30"]
    val_ids = ["R11","R12"]
    test_ids = ["R15"]
    
    counter_seq = 0
    #hist_classes_all = np.zeros((NUM_CLASSES))
    
    for P in persons:
        if usage_modus == 'train':
            recordings = train_ids
        elif usage_modus == 'val':
            recordings = val_ids
        elif usage_modus == 'test':
            recordings = test_ids 
        print("\nModus {} \n{}".format(usage_modus, recordings))
        for R in recordings:
               try:
                    S = SCENARIO[R]
                    file_name_data = "{}/{}_{}_{}.csv".format(P, S, P, R)
                    file_name_label = "{}/{}_{}_{}_labels.csv".format(P, S, P, R)
                    print("\n{}\n{}".format(file_name_data, file_name_label))
                    try:
                        print("check 1")
                        data = read_extracted_data(FOLDER_PATH + file_name_data, skiprows=1)
                        print("\nFiles loaded in modus {}\n{}".format(usage_modus, file_name_data))
                        #data_x = data["data"]
                        data_x = data[:, 2:]
                        print("\nFiles loaded")
                    except:
                        print("\n1 In loading data, in file {}".format(FOLDER_PATH + file_name_data))
                        continue
                    try:
                        # Getting labels and attributes
                        labels = csv_reader.reader_labels(FOLDER_PATH  + file_name_label)
                        print("\n lables loaded")
                        class_labels = np.where(labels[:, 0] == 7)[0]

                        # Deleting rows containing the "none" class
                        data_x = np.delete(data_x, class_labels, 0)
                        labels = np.delete(labels, class_labels, 0)
                    except:
                        print(
                            "2 In generating data, Error getting the data {}".format(FOLDER_PATH 
                                                                                       + file_name_data))
                        continue
                   
                    try:
                        data_x = norm_motion_miners(data_x)
                        print("\n norm done")
                        
                    except:
                        print("\n3  In generating data, Plotting {}".format(FOLDER_PATH + file_name_data))
                        continue
                    try:
                        # checking if annotations are consistent
                        if data_x.shape[0] == data_x.shape[0]:
                            # Sliding window approach
                            print("\nStarting sliding window")
                            X, y, y_all = opp_sliding_window(data_x, labels.astype(int), sliding_window_length,
                                                             sliding_window_step, label_pos_end=False)
                            print("\nWindows are extracted")
                            
                            for f in range(X.shape[0]):
                                try:
                                    sys.stdout.write(
                                        '\r' +
                                        'Creating sequence file number {} with id {}'.format(f, counter_seq))
                                    sys.stdout.flush()

                                    # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                                    seq = np.reshape(X[f], newshape=(1, X.shape[1], X.shape[2]))
                                    seq = np.require(seq, dtype=np.float)

                                    obj = {"data": seq, "act_label": y[f],"label": ID[P]}
                                    file_name = open(os.path.join(data_dir,
                                                                  'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                                    pickle.dump(obj, file_name, protocol=pickle.HIGHEST_PROTOCOL)
                                    file_name.close()

                                    counter_seq += 1
                                    
                                    sys.stdout.write(
                                        '\r' +
                                        'Creating sequence file number {} with id {}'.format(f, counter_seq))
                                    sys.stdout.flush()

                                except:
                                    raise ('\nError adding the seq')

                            print("\nCorrect data extraction from {}".format(FOLDER_PATH + file_name_data))

                            del data
                            del data_x
                            del X
                            del labels
                           
                        else:
                            print("\n4 Not consisting annotation in  {}".format(file_name_data))
                            continue
                    except:
                        print("\n5 In generating data, No created file {}".format(FOLDER_PATH + file_name_data))
                    print("-----------------\n{}\n-----------------".format(file_name_data))
               except KeyboardInterrupt:
                    print('\nYou cancelled the operation.')
                    
    return   

##################

def generate_CSV(csv_dir, type_file, data_dir):
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for n in range(len(filenames)):
            f.append(data_dir + 'seq_{0:06}.pkl'.format(n))

    np.savetxt(csv_dir + type_file, f, delimiter="\n", fmt='%s')
        
    return f

###########################
          
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

    return f

##################################                    
                    
def create_dataset():
 #type1-avoiding person 12
    '''
    train_ids = ["R03", "R07", "R08", "R10", "R11"]
    val_ids = ["R12"]
    test_ids = ["R15"]
    '''
    #type2- avoiding person 11
    '''
    train_ids = ["R11", "R12", "R15", "R18", "R19", "R21"]
    val_ids = ["R22"]
    test_ids = ["R23"]
    '''
    
    #type3- Avoiding person 11 and 12
    '''
    train_ids = ["R03", "R07", "R08", "R10" "R11", "R12", "R15", "R18"]
    val_ids = ["R19", "R21"]
    test_ids = ["R22", "R23"]
    '''
    '''
    #type4-Avoiding persons 11,12,10
    
    train_ids = ["R03", "R07", "R08", "R10" "R11", "R12", "R15", "R18", "R19", "R21", "R22"]
    val_ids = ["R23","R25", "R26"]
    test_ids = ["R27", "R28", "R29"]
    '''
    train_ids = ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", 
                 "R13", "R14", "R16", "R17", "R18", "R19", "R20", "R21", "R22", "R23", 
                 "R24", "R25", "R26", "R27", "R28", "R29", "R30"]
    val_ids = ["R11","R12"]
    test_ids = ["R15"]
    
    
    base_directory='/data/nnair/momin/attr/no6/'
    
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    
    generate_data(train_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_train, usage_modus='train')
    generate_data(val_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_val, usage_modus='val')
    generate_data(test_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_test, usage_modus='test')
    
    generate_CSV(base_directory, "train.csv", data_dir_train)
    generate_CSV(base_directory, "val.csv", data_dir_val)
    generate_CSV(base_directory, "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)

    return


if __name__ == '__main__':
    create_dataset()
    print("Done")