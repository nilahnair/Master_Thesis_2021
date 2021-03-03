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

if __name__ == '__main__':
    
    print("Hello World")
    