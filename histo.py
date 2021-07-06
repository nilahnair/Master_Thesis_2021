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
       print("file opened") 
       # reading the CSV file
       csvFile = csv.DictReader(file)
      
       for lines in csvFile:
           #print(lines)
           for key, values in lines.items():
               if key == 'label':
                   for i in values:
                       inter=int(i) 
                   label.append(inter)
               if key == 'pred':
                   a=values
                   b=[float(i) for i in values[1:-1].split(' ') if i!='']
                   pred.append(b)
                   
                   '''
                   print(a.ndim)
                   print(a.shape)
                   print(a.size)
                   print(a.dtype)
                   '''
                   #pred.append(values)
                   #print("pred")
                   #print(pred)

print(len(label))
print(len(pred))

NUM_CLASSES = 8

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

'''

for i in range(len(label)):
    #print(i)
    if label[i] == 0:
        #print("0")
        k=pred[i]
        for j in range(len(k)):
            counter0.append(k[j])
    elif label[i] == 1:
        #print("1")
        k=pred[i]
        for j in range(len(k)):
            counter1.append(k[j])
    elif label[i] == 2:
        k=pred[i]
        for j in range(len(k)):
            counter2.append(k[j])
    elif label[i] == 3:
        #print("3")
        k=pred[i]
        for j in range(len(k)):
            counter3.append(k[j])
    elif label[i] == 4:
        #print("4")
        k=pred[i]
        for j in range(len(k)):
            counter4.append(k[j])
    elif label[i] == 5:
        #print("5")
        k=pred[i]
        for j in range(len(k)):
            counter5.append(k[j])
    elif label[i] == 6:
        #print("6")
        k=pred[i]
        for j in range(len(k)):
            counter6.append(k[j])
    elif label[i] == 7:
        #print("7")
        k=pred[i]
        for j in range(len(k)):
            k=pred[i]
            counter7.append(k[j])
'''

for i in range(len(label)):
    k=pred[i]
    #print(k)
    for j in range(len(k)):
        if j==0:
            if (label[i] == 0) and (np.argmax(k)==0):
                counterp0.append(k[j])
                indxp0.append(i)
            elif (label[i] == 0) and (np.argmax(k) !=0):
                countern0.append(k[j])
                indxn0.append(i)
        elif j==1:
            if (label[i] == 1) and (np.argmax(k)==1):
                counterp1.append(k[j])
                indxp1.append(i)
            elif (label[i] == 1) and (np.argmax(k)!=1):
                countern1.append(k[j])
                indxn1.append(i)
        elif j==2:
            if (label[i] == 2) and (np.argmax(k)==2):
                counterp2.append(k[j])
                indxp2.append(i)
            elif (label[i] == 2) and (np.argmax(k)!=2):
                countern2.append(k[j])
                indxn2.append(i)
        elif j==3:
            if (label[i] == 3) and (np.argmax(k)==3):
                counterp3.append(k[j])
                indxp3.append(i)
            elif (label[i] == 3) and (np.argmax(k)!=3):
                countern3.append(k[j])
                indxn3.append(i)
        elif j==4:
            if (label[i] == 4) and (np.argmax(k)==4):
                counterp4.append(k[j])
                indxp4.append(i)
            elif (label[i] == 4) and (np.argmax(k)==4):
                countern4.append(k[j])
                indxn4.append(i)
        elif j==5:
            if (label[i] == 5) and (np.argmax(k)==5):
                counterp5.append(k[j])
                indxp5.append(i)
            elif (label[i] == 5) and (np.argmax(k)==5):
                countern5.append(k[j])
                indxn5.append(i)
        elif j==6:    
            if (label[i] == 6) and (np.argmax(k)==6):
                counterp6.append(k[j])
                indxp6.append(i)
            elif (label[i] == 6) and (np.argmax(k)==6):
                countern6.append(k[j])
                indxn6.append(i)
        elif j==7:
            if (label[i] == 7) and (np.argmax(k)==7):
                counterp7.append(k[j])
                indxp7.append(i)
            elif (label[i] == 7) and (np.argmax(k)==7):
                countern7.append(k[j])
                indxn7.append(i)
            
            

            
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
  
#fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7), (ax8, ax9), (ax10, ax11), (ax12, ax13), (ax14, ax15)) = plt.subplots(nrows=8, ncols=2)
'''
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

plt.hist(countern3, bins = 10)
#plt.set_title('counter0 pos')

print(countern3)
print(indxn3)


#fig.tight_layout()
plt.show()
       
