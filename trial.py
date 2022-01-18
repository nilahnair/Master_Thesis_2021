# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 21:56:46 2021

@author: nilah
"""

import numpy as np
import csv
import os
import sys
import matplotlib.pyplot as plt
import datetime
#import csv_reader
#from sliding_window import sliding_window
import pickle
#from torch.utils.data import DataLoader
#from HARWindows import HARWindows
#import torch
#import torch.nn as nn
#from network_act import Network
# load numpy array from csv file
from numpy import loadtxt
# load array

relevance = loadtxt('relevance225.csv', delimiter=',')
print(relevance.shape)
input_d = loadtxt('input225.csv', delimiter=',')
print(input_d.shape)

output = relevance*input_d
print(output.shape)

maxElement = np.amax(relevance, axis=0)
#print("max")
#print(maxElement)
x=range(100)

parameters = {'axes.labelsize': 13}
plt.rcParams.update(parameters)

#plt.plot(x,relevance[:,53], label = "L_toe_TZ")
#plt.plot(x,relevance[:,107], label = "R_toe_TZ")
'''
plt.plot(x, relevance[:,0], label = "head_RX")
plt.plot(x,relevance[:,1], label = "head_RY")
plt.plot(x,relevance[:,2], label = "head_RZ")
plt.plot(x,relevance[:,3], label = "head_TX")
plt.plot(x,relevance[:,4], label = "head_TY")
plt.plot(x,relevance[:,5], label = "head_TZ")
'''
'''
plt.plot(x, relevance[:,54], label = "L_wrist_RX")
plt.plot(x,relevance[:,55], label = "L_wrist_RY")
plt.plot(x,relevance[:,56], label = "L_wrist_RZ")
plt.plot(x,relevance[:,57], label = "L_wrist_TX")
plt.plot(x,relevance[:,58], label = "L_wrist_TY")
plt.plot(x,relevance[:,59], label = "L_wrist_TZ")
'''
#k= relevance[:,6:9]
#print(k)
#rmsav= np.std(k, axis=1)
#print (rmsav)
'''
plt.plot(x,relevance[:,0], label = "AccX_LA")
plt.plot(x,relevance[:,1], label = "AccY_LA")
plt.plot(x,relevance[:,2], label = "AccZ_LA")
'''
'''
plt.plot(x,relevance[:,3], label = "GyrX_LA")
plt.plot(x,relevance[:,4], label = "GyrY_LA")
plt.plot(x,relevance[:,5], label = "GYrZ_LA")
'''
'''
plt.plot(x,relevance[:,6], label = "AccX_LL")
plt.plot(x,relevance[:,7], label = "AccY_LL")
plt.plot(x,relevance[:,8], label = "AccZ_LL")
'''
'''
plt.plot(x,relevance[:,9], label = "GyrX_LL")
plt.plot(x,relevance[:,10], label = "GyrY_LL")
plt.plot(x,relevance[:,11], label = "GYrZ_LL")
'''
'''
plt.plot(x,relevance[:,12], label = "AccX_N")
plt.plot(x,relevance[:,13], label = "AccY_N")
plt.plot(x,relevance[:,14], label = "AccZ_N")
'''
'''
plt.plot(x,relevance[:,15], label = "GyrX_N")
plt.plot(x,relevance[:,16], label = "GyrY_N")
plt.plot(x,relevance[:,17], label = "GYrZ_N")
'''
'''
plt.plot(x,relevance[:,18], label = "AccX_RA")
plt.plot(x,relevance[:,19], label = "AccY_RA")
plt.plot(x,relevance[:,20], label = "AccZ_RA")
'''
'''
plt.plot(x,relevance[:,21], label = "GyrX_RA")
plt.plot(x,relevance[:,22], label = "GyrY_RA")
plt.plot(x,relevance[:,23], label = "GYrZ_RA")
'''
'''
plt.plot(x,relevance[:,24], label = "AccX_RL")
plt.plot(x,relevance[:,25], label = "AccY_RL")
plt.plot(x,relevance[:,26], label = "AccZ_RL")
'''
'''
plt.plot(x,relevance[:,27], label = "GyrX_RL")
plt.plot(x,relevance[:,28], label = "GyrY_RL")
plt.plot(x,relevance[:,29], label = "GYrZ_RL")
'''
#print(relevance[:,3:6])
#print(np.where(relevance[:,3:6]>0,relevance[:,3:6],0))

plt.plot(x,np.sqrt(np.mean(np.square(np.where(relevance[:,0:6]>0,relevance[:,0:6],0)), axis=1)), label = "LA")
plt.plot(x,np.sqrt(np.mean(np.square(np.where(relevance[:,6:12]>0,relevance[:,6:12],0)), axis=1)), label = "LL")
plt.plot(x,np.sqrt(np.mean(np.square(np.where(relevance[:,12:18]>0,relevance[:,12:18],0)), axis=1)), label = "N")
plt.plot(x,np.sqrt(np.mean(np.square(np.where(relevance[:,18:24]>0,relevance[:,18:24],0)), axis=1)), label = "RA")
plt.plot(x,np.sqrt(np.mean(np.square(np.where(relevance[:,24:30]>0,relevance[:,24:30],0)), axis=1)), label = "RL")


'''
plt.plot(x,np.sqrt(np.mean(np.square(relevance[:,3:6]), axis=1)), label = "GYR_LA")
plt.plot(x,np.sqrt(np.mean(np.square(relevance[:,9:12]), axis=1)), label = "GYR_LL")
plt.plot(x,np.sqrt(np.mean(np.square(relevance[:,15:18]), axis=1)), label = "GYR_N")
plt.plot(x,np.sqrt(np.mean(np.square(relevance[:,21:24]), axis=1)), label = "GYR_RA")
plt.plot(x,np.sqrt(np.mean(np.square(relevance[:,27:30]), axis=1)), label = "GYR_RL")
'''
'''
idx_LA = np.arange(12, 24)
idx_LA = np.concatenate([idx_LA, np.arange(36, 42)])
idx_LA = np.concatenate([idx_LA, np.arange(54, 66)])
LA=relevance[:,idx_LA]

idx_LL = np.arange(24, 36)
idx_LL = np.concatenate([idx_LL, np.arange(42, 54)])
LL=relevance[:,idx_LL]
                    
idx_N = np.arange(0, 12)
idx_N = np.concatenate([idx_N, np.arange(120, 126)])
N= relevance[:,idx_N]

idx_RA = np.arange(66, 78)
idx_RA = np.concatenate([idx_RA, np.arange(90, 96)])
idx_RA = np.concatenate([idx_RA, np.arange(108, 120)])
RA=relevance[:,idx_RA]    

idx_RL = np.arange(78, 90)
idx_RL = np.concatenate([idx_RL, np.arange(96, 108)])
RL=relevance[:,idx_RL]                         

plt.plot(x,np.sqrt(np.mean(np.square(np.where(LA>0,LA,0)), axis=1)), label = "LA")
plt.plot(x,np.sqrt(np.mean(np.square(np.where(LL>0,LL,0)), axis=1)), label = "LL")
plt.plot(x,np.sqrt(np.mean(np.square(np.where(N>0,N,0)), axis=1)), label = "N")
plt.plot(x,np.sqrt(np.mean(np.square(np.where(RA>0,RA,0)), axis=1)), label = "RA")
plt.plot(x,np.sqrt(np.mean(np.square(np.where(RL>0,RL,0)), axis=1)), label = "RL")
'''
#print(relevance[:,3:6])
#print(np.sqrt(np.mean(np.square(relevance[:,3:6]), axis=1)))
#plt.plot(x,rmsav, label = "RMS")
#plt.plot(x,relevance)
plt.xlabel("Time frame")
plt.ylabel("RMS Relevance")
plt.title('Subject 0, Cart activity, Mbientlab')
#leg = plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.15),ncol=2)
leg = plt.legend(loc='upper right')
plt.show()




'''
with np.load("../Master_Thesis_2021/test_imu.npz") as data:
    d2=data['d']
    l2=data['l']
    al2=data['al']
    p2=data['p']
    
print(d2.shape)
print(l2)
print(al2.shape)
print(p2.shape)

counterp0=[]
counterp1=[]
counterp2=[]
counterp3=[]
counterp4=[]
counterp5=[]
counterp6=[]
counterp7=[]
countern0=[]
countern1=[]
countern2=[]
countern3=[]
countern4=[]
countern5=[]
countern6=[]
countern7=[]

indxp0=[]
indxp1=[]
indxp2=[]
indxp3=[]
indxp4=[]
indxp5=[]
indxp6=[]
indxp7=[]
indxn0=[]
indxn1=[]
indxn2=[]
indxn3=[]
indxn4=[]
indxn5=[]
indxn6=[]
indxn7=[]

for i in range(len(l2)):
    k=p2[i]
    #print(k)
    for j in range(len(k)):
        if j==0:
            if (l2[i] == 0) and (np.argmax(k)==0):
                counterp0.append(k[j])
                indxp0.append(i)
            elif (l2[i] == 0) and (np.argmax(k) !=0):
                countern0.append(k[j])
                indxn0.append(i)
        elif j==1:
            if (l2[i] == 1) and (np.argmax(k)==1):
                counterp1.append(k[j])
                indxp1.append(i)
            elif (l2[i] == 1) and (np.argmax(k)!=1):
                countern1.append(k[j])
                indxn1.append(i)
        elif j==2:
            if (l2[i] == 2) and (np.argmax(k)==2):
                counterp2.append(k[j])
                indxp2.append(i)
            elif (l2[i] == 2) and (np.argmax(k)!=2):
                countern2.append(k[j])
                indxn2.append(i)
        elif j==3:
            if (l2[i] == 3) and (np.argmax(k)==3):
                counterp3.append(k[j])
                indxp3.append(i)
            elif (l2[i] == 3) and (np.argmax(k)!=3):
                countern3.append(k[j])
                indxn3.append(i)
        elif j==4:
            if (l2[i] == 4) and (np.argmax(k)==4):
                counterp4.append(k[j])
                indxp4.append(i)
            elif (l2[i] == 4) and (np.argmax(k)==4):
                countern4.append(k[j])
                indxn4.append(i)
        elif j==5:
            if (l2[i] == 5) and (np.argmax(k)==5):
                counterp5.append(k[j])
                indxp5.append(i)
            elif (l2[i] == 5) and (np.argmax(k)==5):
                countern5.append(k[j])
                indxn5.append(i)
        elif j==6:    
            if (l2[i] == 6) and (np.argmax(k)==6):
                counterp6.append(k[j])
                indxp6.append(i)
            elif (l2[i] == 6) and (np.argmax(k)==6):
                countern6.append(k[j])
                indxn6.append(i)
        elif j==7:
            if (l2[i] == 7) and (np.argmax(k)==7):
                counterp7.append(k[j])
                indxp7.append(i)
            elif (l2[i] == 7) and (np.argmax(k)==7):
                countern7.append(k[j])
                indxn7.append(i)
            
            
'''
            
'''
print(len(counter0))
print(len(counter1))
print(len(counter2))
print(len(counter3))
print(len(counter4))
print(len(counter5))
print(len(counter6))
print(len(counter7))
'''
  
'''
fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7), (ax8, ax9), (ax10, ax11), (ax12, ax13), (ax14, ax15)) = plt.subplots(nrows=8, ncols=2)

ax0.hist(counterp0, bins = 10)
ax0.set_title('counter0 pos')

ax1.hist(countern0, bins = 10)
ax1.set_title('counter0 neg')

ax2.hist(counterp1, bins = 10)
ax2.set_title('counter1 pos')

ax3.hist(countern1, bins = 10)
ax3.set_title('counter1 neg')

ax4.hist(counterp2, bins = 10)
ax4.set_title('counter2 pos')

ax5.hist(countern2, bins = 10)
ax5.set_title('counter2 neg')

ax6.hist(counterp3, bins = 10)
ax6.set_title('counter3 pos')

ax7.hist(countern3, bins = 10)
ax7.set_title('counter3 neg')

ax8.hist(counterp4, bins = 10)
ax8.set_title('counter4 pos')

ax9.hist(countern4, bins = 10)
ax9.set_title('counter4 neg')

ax10.hist(counterp5, bins = 10)
ax10.set_title('counter5 pos')

ax11.hist(countern5, bins = 10)
ax11.set_title('counter5 neg')

ax12.hist(counterp6, bins = 10)
ax12.set_title('counter6 pos')

ax13.hist(countern6, bins = 10)
ax13.set_title('counter6 neg')

ax14.hist(counterp7, bins = 10)
ax14.set_title('counter7 pos')

ax15.hist(countern7, bins = 10)
ax15.set_title('counter7 neg')
'''

'''
plt.hist(counterp3, bins = 10)
#plt.set_title('counter0 pos')

print(counterp3)
print(indxp3)
'''

#fig.tight_layout()
#plt.show()

#plt.savefig("imu.png")
'''
network_obj = Network(self.config)
model.load_state_dict(torch.load(PATH))
model.eval()
'''

'''
import torch
torch.cuda.empty_cache()

torch.cuda.memory_summary(device=None, abbreviated=False)
'''

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