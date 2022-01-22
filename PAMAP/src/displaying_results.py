'''
Created on March 13, 2021

@author: fmoya
'''

from __future__ import print_function
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def displaying_results(path):

    tree = ET.parse(path + "results_yy2022mm1dd10hh09mm17.xml")
    root = tree.getroot()

    print(root.tag)
    print(root.attrib)

    for child in root:
        print(child.tag, child.attrib)

    print(root[-2][0].attrib)
    print(root[-2][0][0].attrib)
    print(root[-2][0][1].attrib)
    print(root[-2][0][2].attrib)
    print(root[-2][0][3].attrib)
    print(root[-2][0][4].attrib)
    print(root[-2][0][5].attrib)
    print(root[-2][0][6].attrib)
    print(root[-2][0][7].attrib)
    print(root[-2][0][8].attrib)
    print(root[-2][0][9].attrib)
    print(root[-2][0][10].attrib)
    print(root[-2][0][10].attrib)
    print(root[-2][0][11].attrib)
    print("\n")

    train_id = 19  #or 15 or 19
    plus_seg = 10 #or 6 or 10

    # #Val
    print("Mean Val")#14
    print(root[-2][0][train_id].attrib['acc_test_mean'])
    print(root[-2][0][train_id].attrib['f1_mean_test_mean'])
    print(root[-2][0][train_id].attrib['f1_weighted_test_mean'])

    print("Std Val")
    print(root[-2][0][train_id + 1].attrib['acc_test_std'])
    print(root[-2][0][train_id + 1].attrib['f1_mean_test_std'])
    print(root[-2][0][train_id + 1].attrib['f1_weighted_test_std'])

    #Val Segmentation
    print("Mean Val Segmentation")
    print(root[-2][0][train_id + plus_seg].attrib['acc_test_seg_mean'])
    print(root[-2][0][train_id + plus_seg].attrib['f1_mean_test_seg_mean'])
    print(root[-2][0][train_id + plus_seg].attrib['f1_weighted_test_seg_mean'])

    print("Std Test Segmentation")
    print(root[-2][0][train_id + plus_seg + 1].attrib['acc_test_seg_std'])
    print(root[-2][0][train_id + plus_seg + 1].attrib['f1_mean_test_seg_std'])
    print(root[-2][0][train_id + plus_seg + 1].attrib['f1_weighted_test_seg_std'])

    plus_seg_test = 12  # or 8 or 12

    #Test
    print("Mean Test")
    print(root[-1][0][train_id].attrib['acc_test_mean'])
    print(root[-1][0][train_id].attrib['f1_mean_test_mean'])
    print(root[-1][0][train_id].attrib['f1_weighted_test_mean'])

    print("Std Test")
    print(root[-1][0][train_id + 1].attrib['acc_test_std'])
    print(root[-1][0][train_id + 1].attrib['f1_mean_test_std'])
    print(root[-1][0][train_id + 1].attrib['f1_weighted_test_std'])

    #Test
    print("Mean Test Segmentation")
    print(root[-1][0][train_id + plus_seg_test].attrib['acc_test_seg_mean'])
    print(root[-1][0][train_id + plus_seg_test].attrib['f1_mean_test_seg_mean'])
    print(root[-1][0][train_id + plus_seg_test].attrib['f1_weighted_test_seg_mean'])

    print("Std Test Segmentation")
    print(root[-1][0][train_id + plus_seg_test + 1].attrib['acc_test_seg_std'])
    print(root[-1][0][train_id + plus_seg_test + 1].attrib['f1_mean_test_seg_std'])
    print(root[-1][0][train_id + plus_seg_test + 1].attrib['f1_weighted_test_seg_std'])

    #Precision mean
    test_id = 22 # or 18 or 22

    print("Precision mean Test") #18
    for pr in (root[-1][0][test_id].attrib['precision_mean'][1:-1]).split("0.")[1:]:
        print("0." + pr)

    #Precision std
    print("Precision std Test")
    for pr in (root[-1][0][test_id + 1].attrib['precision_std'][1:-1]).split("0.")[1:]:
        print("0." + pr)

    #Recall mean
    print("Recall mean Test")
    for pr in (root[-1][0][test_id + 2].attrib['recall_mean'][1:-1]).split("0.")[1:]:
        print("0." + pr)

    #Recall std
    print("Recall std Test")
    for pr in (root[-1][0][test_id + 3].attrib['recall_std'][1:-1]).split("0.")[1:]:
        print("0." + pr)

    return


def list_files(pathfile):

    for root_d, dirs_D, files in os.walk(pathfile):
        for f in files:
            print(f)

    return

if __name__ == '__main__':

    #pathfile = "/data2/fmoya/HAR/pytorch/gesture/cnn_imu/attribute/nopooling/LSTM/noreshape/experiment/"
    #pathfile = "/data2/fmoya/HAR/pytorch/gesture/cnn_imu/attribute/nopooling/LSTM/noreshape/experiment/"
    pathfile = "/data/fmoya/HAR/pytorch/pamap2/cnn/softmax/nopooling/FCN/noreshape/fine_tuning/"

    #list_files(pathfile)
    displaying_results(pathfile)

    print("Done")

