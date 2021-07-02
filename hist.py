# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:24:49 2021

@author: nilah
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

data=[]
label=[]
pred=[]
with open('../Master_Thesis_2021/test_files.csv', mode ='r') as file:   
        
       # reading the CSV file
       csvFile = csv.DictReader(file)
       for lines in csvFile:
           #print(lines)
           for key, values in lines.items():
               if key == 'label':
                   label.append(values)
               if key == 'pred':
                   #a=np.array(values)
                   print(eval(values))
                   pred.append(values)
                   
print(len(label))
print(len(pred))
pred=np.array(pred)
#print(pred)
print()
NUM_CLASSES = 8

counter0=[]
counter1=[]
counter2=[]
counter3=[]
counter4=[]
counter5=[]
counter6=[]
counter7=[]

    

for i in range(len(label)):
    print(type(label[i]))
    if label[i] == '0':
        print(type(pred[i,:]))
        print(pred[i,:])
        print(type(pred[i,:]))
        counter0.append(int(pred[i,:]))
    elif label[i] == '1':
        counter1.append(int(pred[i,:]))
    elif label[i] == '2':
        counter2.append(int(pred[i,:]))
    elif label[i] == '3':
        counter3.append(int(pred[i,:]))
    elif label[i] == '4':
        counter4.append(int(pred[i,:]))
    elif label[i] == '5':
        counter5.append(pred[i,:])
    elif label[i] == '6':
        counter6.append(pred[i,:])
    elif label[i] == '7':
        counter7.append(pred[i,:])

print(counter0)
print(counter1)
print(len(counter2))
print(len(counter3))
print(len(counter4))
print(len(counter5))
print(len(counter6))
print(len(counter7))
    
        


    
hist_classes_all = np.zeros((NUM_CLASSES))
       
