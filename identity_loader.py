# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 18:15:22 2021

@author: nilah
Taken from fernando_moya - https://github.com/wilfer9008/Annotation_Tool_LARa/tree/master/From_Human_Pose_to_On_Body_Devices_for_Human_Activity_Recognition/LARA_dataset
"""

import os

from torch.utils.data import Dataset


import pandas as pd
import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class (Dataset):
    '''
    classdocs
    '''


    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with list of annotated sequences.
            root_dir (string): Directory with all the sequences.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.identity_loader = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.identity_loader)

    def __getitem__(self, idx):
        '''
        get single item

        @param data: index of item in List
        @return window_data: dict with sequence window, label of window, and labels of each sample in window
        '''
        window_name = os.path.join(self.root_dir, self.identity_loader.iloc[idx, 0])

        f = open(window_name, 'rb')
        data = pickle.load(f, encoding='bytes')
        f.close()

        X = data['data']
        y = data['label']
        
               
        window_data = {"data": X, "label": y}

        return window_data
