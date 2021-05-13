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
from torch.utils.data import DataLoader
from HARWindows import HARWindows
'''
import torch
torch.cuda.empty_cache()

'''
path="/home/nnair/Master_Thesis_2021/id_attr_one.txt"
att_rep = np.loadtxt(path, delimiter=',')
print(att_rep)
print(att_rep.shape)
for i in range(0,att_rep.shape[0]):
    print(att_rep[i,1:])
    a= att_rep[i,1:]
    
  
sample=[1, 3, 0, 2, 4, 7, 6, 1, 2, 5, 2, 7, 5, 0, 3, 0, 4]   
sample=np.asarray(sample)
train_batch_l=np.zeros((sample.shape[0],att_rep.shape[1]-1)) 
for i in range(sample.shape[0]):
    if sample[i]==att_rep[sample[i],0]:
        train_batch_l[i]= att_rep[sample[i],1:]
    
print(train_batch_l)
'''


'''
ws=5
ss=1
data_y = np.ones( (30,20) , dtype=np.int64)
print(data_y)
data_y=data_y[1]
print(data_y)
'''
'''
root='/data/nnair/output/type1/mocap/'
root1='/data/nnair/output/type1/mocap/train.csv'
print("check1")

harwindows_train = HARWindows(csv_file = root1 , root_dir=root)
print("check2")

dataLoader_train = DataLoader(harwindows_train, batch_size=100, shuffle=True)
print(dataLoader_train)
len(dataLoader_train)
itera=1
for b, harwindow_batched in enumerate(dataLoader_train):
                sys.stdout.write("\rTraining: Epoch {}/{} Batch {}/{} and itera {}".format(1,
                                                                                          1,
                                                                                           1,
                                                                                           len(dataLoader_train),
                                                                                           itera))
                sys.stdout.flush()

                #Setting the network to train mode
                #network_obj.train(mode=True)
                
                #Counting iterations
                itera = (1 * harwindow_batched["data"].shape[0]) + 1
                print(itera)
                #Selecting batch
                train_batch_v = harwindow_batched["data"]
                print(train_batch_v)
                train_batch_l = harwindow_batched["label"][:, 0]
                print(train_batch_l)
                train_batch_l = train_batch_l.reshape(-1)
                print(train_batch_l)
        
'''
'''
def generate_CSV(csv_dir, data_dir):
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for n in range(len(filenames)):
            f.append(data_dir + 'seq_{0:06}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')
        
    return f

if __name__ == '__main__':
    
    base_directory='/data/nnair/output/type1/imu_norm/'
    
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
 
    generate_CSV(base_directory + "train.csv", data_dir_train)
    generate_CSV(base_directory + "val.csv", data_dir_val)
    generate_CSV(base_directory + "test.csv", data_dir_test)
'''
'''
path="C:/Users/nilah/Desktop/German/Master thesis basis/L01_S07_R01.csv"
IMU = []
time = []
data = []
accumulator_measurements = np.empty((0, 30))
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

    if len(row) != 31:
        imu_data = {'IMU': IMU, 'time': time, 'data': data}
    else:
        imu_data = {'time': time, 'data': data}
data_new= imu_data["data"]
data_new=np.asarray(data_new)
data_x = data_new
print(data_x.shape)
accumulator_measurements = np.append(accumulator_measurements, data_x, axis=0)
print(len(data_x))

data_x=np.asarray(data_x)
print(data_x.shape)
print(data_x.shape[0] == data_x.shape[0])
data_norm = sliding_window(data_x, (100, data_x.shape[1]), (12, 1))
print(data_norm[:,])
label=1
data_y=np.full(data_norm.shape[0],label)
'''
'''
    path="C:/Users/nilah/Desktop/German/wrok revision/States/L01_S01_R01_A17_N01_norm_data.csv"
    with open(path, 'r') as csvfile:
       spamreader =csv.reader(csvfile, delimiter=',', quotechar='|')
       for row in spamreader:
           if spamreader.line_num == 1:
               print(row)
'''
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
        print(path1)
        for row in spamreader:
           if spamreader.line_num == 1:
               print(row)
    
    path2="/vol/actrec/DFG_Project/2019/LARa_dataset/MoCap/recordings_2019/14_Annotated_Dataset_renamed/S07/L01_S07_R01_A03_N02_norm_data.csv"
    
    with open(path2, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        print("Eg: mocap data")
        print(path2)
        for row in spamreader:
           if spamreader.line_num == 1:
               print(row)
    
    path3="/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/flw_recordings_annotated/S07/L01_S07_R01.csv"
    
    with open(path3, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        print("Eg: motionminer data")
        print(path3)
        for row in spamreader:
           if spamreader.line_num == 1:
               print(row)
    
    base_directory='/data/nnair/trial/'
    
    data_dir_train = base_directory + 'sequences_train/'
    
    csv_dir= data_dir_train+"train_final.csv"
    print(csv_dir)
    x= np.arange(0.0,5.0,1.0)
    np.savetxt(csv_dir, x, delimiter="\n", fmt='%s')
  
Max_values=[5.35345435, 9.10256798, 4.88436873, 0.00138411284, 0.00138515507, 0.0598900912, 0.106352326, 6.80991485, 4.80592809, 0.0152394273, 0.0739000245, 0.0312218774, 4.20999100, 1.88736763, 4.04904100, 3.63478717, 5.83114797, 7.33353379, 5.86790924, 2.62255038, 6.17225572, 7.77781943, 2.16467766, 4.55743908, 1.01340060, 5.08785669, 3.51806116, 3.13018543, 1.50726462, 6.16330964]
Mean_values=[5.35345435, 9.10256798, 4.88436873, 0.00138411284, 0.00138515507, 0.0598900912, 0.106352326, 6.80991485, 4.80592809, 0.0152394273, 0.0739000245, 0.0312218774, 4.20999100, 1.88736763, 4.04904100, 3.63478717, 5.83114797, 7.33353379, 5.86790924, 2.62255038, 6.17225572, 7.77781943, 2.16467766, 4.55743908, 1.01340060, 5.08785669, 3.51806116, 3.13018543, 1.50726462, 6.16330964]
Min_values=[5.35345435, 9.10256798, 4.88436873, 0.00138411284, 0.00138515507, 0.0598900912, 0.106352326, 6.80991485, 4.80592809, 0.0152394273, 0.0739000245, 0.0312218774, 4.20999100, 1.88736763, 4.04904100, 3.63478717, 5.83114797, 7.33353379, 5.86790924, 2.62255038, 6.17225572, 7.77781943, 2.16467766, 4.55743908, 1.01340060, 5.08785669, 3.51806116, 3.13018543, 1.50726462, 6.16330964]
x=[]
x.append(Max_values)
x.append(Min_values)
x.append(Mean_values)
x=np.asarray(x)
'''