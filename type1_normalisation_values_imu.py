# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:26:32 2021

@author: nilah
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import csv_reader
import datetime


headers_annotated = ['Time', 'Class', 'AccX_L', 'AccY_L', 'AccZ_L', 'GyrX_L', 'GyrY_L', 'GyrZ_L',
           'MagX_L', 'MagY_L', 'MagZ_L', 'AccX_T', 'AccY_T', 'AccZ_T', 'GyrX_T', 'GyrY_T',
           'GyrZ_T', 'MagX_T', 'MagY_T', 'MagZ_T', 'AccX_R', 'AccY_R', 'AccZ_R', 'GyrX_R',
           'GyrY_R', 'GyrZ_R', 'MagX_R', 'MagY_R', 'MagZ_R']

headers = ['Time', 'AccX_L', 'AccY_L', 'AccZ_L', 'GyrX_L', 'GyrY_L', 'GyrZ_L',
           'MagX_L', 'MagY_L', 'MagZ_L', 'AccX_T', 'AccY_T', 'AccZ_T', 'GyrX_T', 'GyrY_T',
           'GyrZ_T', 'MagX_T', 'MagY_T', 'MagZ_T', 'AccX_R', 'AccY_R', 'AccZ_R', 'GyrX_R',
           'GyrY_R', 'GyrZ_R', 'MagX_R', 'MagY_R', 'MagZ_R']

SCENARIO = {'R01': 'L01', 'R02': 'L01', 'R03': 'L02', 'R04': 'L02', 'R05': 'L02', 'R06': 'L02', 'R07': 'L02',
            'R08': 'L02', 'R09': 'L02', 'R10': 'L02', 'R11': 'L02', 'R12': 'L02', 'R13': 'L02', 'R14': 'L02',
            'R15': 'L02', 'R16': 'L02', 'R17': 'L03', 'R18': 'L03', 'R19': 'L03', 'R20': 'L03', 'R21': 'L03',
            'R22': 'L03', 'R23': 'L03', 'R24': 'L03', 'R25': 'L03', 'R26': 'L03', 'R27': 'L03', 'R28': 'L03',
            'R29': 'L03', 'R30': 'L03'}

labels_persons = {"S01": 0, "S02": 1, "S03": 2, "S04": 3, "S05": 4, "S06": 5, "S07": 6, "S08": 7, "S09": 8,
                  "S10": 9, "S11": 10, "S12": 11, "S13": 12, "S14": 13, "S15": 14, "S16": 15}

def read_extracted_data(path, skiprows = 1):
    '''
    gets data from csv file
    data contains 3 columns, start, end and label

    returns a numpy array

    @param path: path to file
    '''

    annotation_original = np.loadtxt(path, delimiter=',', skiprows=skiprows)
    return annotation_original

def statistics_measurements():
    '''
    Computes some statistics over the channels for the entire training data

    returns a max_values, min_values, mean_values, std_values
    '''

    #dataset_path_imu = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/flw_recordings_12000/"
    dataset_path_imu = "/vol/actrec/DFG_Project/2019/LARa_dataset/Mbientlab/LARa_dataset_mbientlab/"

    #train_final_ids = ["S07", "S08", "S09", "S10", "S11", "S12"]

    persons = ["S07", "S08", "S09", "S10", "S12", "S13", "S14"]
    
    #recordings = ['R{:02d}'.format(r) for r in range(1, 31)]
    
    train_ids = ["R11", "R12", "R15", "R18", "R19","R21"]
    #val_ids = ["R12"]
    #test_ids = ["R15"]
    IMU = []
    time = []
    data = []

    accumulator_measurements = np.empty((0, 30))
    for P in persons:
          for R in train_ids:
                S = SCENARIO[R]
                file_name_data = "{}/{}_{}_{}.csv".format(P, S, P, R)
                #file_name_label = "{}/{}_{}_{}_labels.csv".format(P, S, P, R)
                print("------------------------------\n{}".format(file_name_data))
                # getting data
                path=dataset_path_imu + file_name_data
                with open(path, 'r') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                    for row in spamreader:
                        try:
                            try:
                                if spamreader.line_num == 1:
                                    # print('\n')
                                    print(', '.join(row))
                                else:
                                    if len(row) != 31:
                                        idx_row = 0
                                        IMU.append(row[idx_row])
                                        idx_row += 1
                                    else:
                                        idx_row = 0
                                    try:
                                        time_d = datetime.datetime.strptime(row[idx_row], '%Y-%m-%d %H:%M:%S.%f')
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
                    
                    print(len(data[0]))
                    print(len(data[1]))
                                
                    if len(row) != 31:
                        imu_data = {'IMU': IMU, 'time': time, 'data': data}
                    else:
                        try:
                            print("check")
                            imu_data = {'time': time, 'data': data}
                            data_new=np.asarray(data)
                            print(data_new.shape)
                            print(accumulator_measurements.shape)
                            accumulator_measurements = np.append(accumulator_measurements, data_new, axis=0)
                            print("\nFiles loaded")
                        except:
                            print("\n1 In loading data,  in file {}".format(dataset_path_imu + file_name_data))
                            continue
                            
    
    try:
        max_values = np.max(accumulator_measurements, axis=0)
        print("Max values")
        print(max_values)
        min_values = np.min(accumulator_measurements, axis=0)
        print("Min values")
        print(min_values)
        mean_values = np.mean(accumulator_measurements, axis=0)
        print("Mean values")
        print(mean_values)
        std_values = np.std(accumulator_measurements, axis=0)
        print("std values")
        print(std_values)
    except:
        max_values = 0
        min_values = 0
        mean_values = 0
        std_values = 0
        print("Error computing statistics")
    
    return max_values, min_values, mean_values, std_values

if __name__ == '__main__':
    
    #Computing Statistics of data
    max_values, min_values, mean_values, std_values = statistics_measurements()
    
    x = []
    x.append(list(max_values))
    x.append(list(min_values))
    x.append(list(mean_values))
    x.append(list(std_values))
    x=np.asarray(x)
    print(x)
  
    base_directory='/data/nnair/trial/'
    
    csv_dir=  base_directory+"type2_normalisation_values_imu.csv"
    print(csv_dir)
    np.savetxt(csv_dir, x, delimiter="\n", fmt='%s')