# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:24:49 2021

@author: nilah
"""

import csv

with open('../Master_Thesis_2021/test_files.csv', mode ='r') as file:   
        
       # reading the CSV file
       csvFile = csv.DictReader(file)
 
       # displaying the contents of the CSV file
       for lines in csvFile:
            print(lines)
       