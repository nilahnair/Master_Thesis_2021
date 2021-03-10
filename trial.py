# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 21:56:46 2021

@author: nilah
"""
'''
import numpy as np
import csv
import os
import sys
import matplotlib.pyplot as plt
import datetime
#import csv_reader
from sliding_window import sliding_window
import pickle

path="C:/Users/nilah/Desktop/German/Master thesis basis/Master_Thesis_2021/L03_S07_R30.csv"
IMU = []
time = []
data = []
with open(path, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    print(spamreader)
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

    if len(row) != 31:
        imu_data = {'IMU': IMU, 'time': time, 'data': data}
    else:
        imu_data = {'time': time, 'data': data}
data_x = imu_data["data"]
print(len(data_x))

data_x=np.asarray(data_x)
print(data_x.shape)
print(data_x.shape[0] == data_x.shape[0])
data_norm = sliding_window(data_x, (100, data_x.shape[1]), (12, 1))
print(data_norm[:,])
label=1
data_y=np.full(data_norm.shape[0],label)
'''
import numpy as np
import csv
import os
import sys
import pickle

if __name__ == '__main__':
    
    print("Hello World")
    path1="/vol/actrec/DFG_Project/2019/LARa_dataset/Mbientlab/LARa_dataset_mbientlab/S07/L01_S07_R01.csv"
    
    with open(path1, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        print("Eg: ambient data")
        print(spamreader)
    
    path2="/vol/actrec/DFG_Project/2019/LARa_dataset/MoCap/recordings_2019/14_Annotated_Dataset_renamed/S07/L01_S07_R01_A03_N02_norm_data.csv"
    
    with open(path2, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        print("Eg: mocap data")
        print(spamreader)
    
    path3="/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/flw_recordings_annotated/S07/L01_S07_R01.csv"
    
    with open(path3, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        print("Eg: motionminer data")
        print(spamreader)
    
    base_directory='/data/nnair/trial/'
    
    data_dir_train = base_directory + 'sequences_train/'
    
    csv_dir= data_dir_train+"train_final.csv"
    np.savetxt(csv_dir, delimiter="\n", fmt='%s')
    
    