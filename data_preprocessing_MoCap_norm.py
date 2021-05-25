# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:18:36 2021

@author: nilah ravi nair
code adapted from: fernando moya
"""

import os
import sys
import numpy as np

import csv_reader
from sliding_window import sliding_window
import pickle

FOLDER_PATH = "/vol/actrec/DFG_Project/2019/LARa_dataset/MoCap/recordings_2019/14_Annotated_Dataset_renamed/"

NUM_CLASSES=7

headers = ["sample", "label", "head_RX", "head_RY", "head_RZ", "head_TX", "head_TY", "head_TZ", "head_end_RX",
           "head_end_RY", "head_end_RZ", "head_end_TX", "head_end_TY", "head_end_TZ", "L_collar_RX", "L_collar_RY",
           "L_collar_RZ", "L_collar_TX", "L_collar_TY", "L_collar_TZ", "L_elbow_RX", "L_elbow_RY", "L_elbow_RZ",
           "L_elbow_TX", "L_elbow_TY", "L_elbow_TZ", "L_femur_RX", "L_femur_RY", "L_femur_RZ", "L_femur_TX",
           "L_femur_TY", "L_femur_TZ", "L_foot_RX", "L_foot_RY", "L_foot_RZ", "L_foot_TX", "L_foot_TY", "L_foot_TZ",
           "L_humerus_RX", "L_humerus_RY", "L_humerus_RZ", "L_humerus_TX", "L_humerus_TY", "L_humerus_TZ", "L_tibia_RX",
           "L_tibia_RY", "L_tibia_RZ", "L_tibia_TX", "L_tibia_TY", "L_tibia_TZ", "L_toe_RX", "L_toe_RY", "L_toe_RZ",
           "L_toe_TX", "L_toe_TY", "L_toe_TZ", "L_wrist_RX", "L_wrist_RY", "L_wrist_RZ", "L_wrist_TX", "L_wrist_TY",
           "L_wrist_TZ", "L_wrist_end_RX", "L_wrist_end_RY", "L_wrist_end_RZ", "L_wrist_end_TX", "L_wrist_end_TY",
           "L_wrist_end_TZ", "R_collar_RX", "R_collar_RY", "R_collar_RZ", "R_collar_TX", "R_collar_TY", "R_collar_TZ",
           "R_elbow_RX", "R_elbow_RY", "R_elbow_RZ", "R_elbow_TX", "R_elbow_TY", "R_elbow_TZ", "R_femur_RX",
           "R_femur_RY", "R_femur_RZ", "R_femur_TX", "R_femur_TY", "R_femur_TZ", "R_foot_RX", "R_foot_RY", "R_foot_RZ",
           "R_foot_TX", "R_foot_TY", "R_foot_TZ", "R_humerus_RX", "R_humerus_RY", "R_humerus_RZ", "R_humerus_TX",
           "R_humerus_TY", "R_humerus_TZ", "R_tibia_RX", "R_tibia_RY", "R_tibia_RZ", "R_tibia_TX", "R_tibia_TY",
           "R_tibia_TZ", "R_toe_RX", "R_toe_RY", "R_toe_RZ", "R_toe_TX", "R_toe_TY", "R_toe_TZ", "R_wrist_RX",
           "R_wrist_RY", "R_wrist_RZ", "R_wrist_TX", "R_wrist_TY", "R_wrist_TZ", "R_wrist_end_RX", "R_wrist_end_RY",
           "R_wrist_end_RZ", "R_wrist_end_TX", "R_wrist_end_TY", "R_wrist_end_TZ", "root_RX", "root_RY", "root_RZ",
           "root_TX", "root_TY", "root_TZ"]

SCENARIO = {'R01': 'L01', 'R02': 'L01', 'R03': 'L02', 'R04': 'L02', 'R05': 'L02', 'R06': 'L02', 'R07': 'L02',
            'R08': 'L02', 'R09': 'L02', 'R10': 'L02', 'R11': 'L02', 'R12': 'L02', 'R13': 'L02', 'R14': 'L02',
            'R15': 'L02', 'R16': 'L02', 'R17': 'L03', 'R18': 'L03', 'R19': 'L03', 'R20': 'L03', 'R21': 'L03',
            'R22': 'L03', 'R23': 'L03', 'R24': 'L03', 'R25': 'L03', 'R26': 'L03', 'R27': 'L03', 'R28': 'L03',
            'R29': 'L03', 'R30': 'L03'}

annotator = {"S01": "A17", "S02": "A03", "S03": "A08", "S04": "A06", "S05": "A12", "S06": "A13",
             "S07": "A05", "S08": "A17", "S09": "A03", "S10": "A18", "S11": "A08", "S12": "A11",
             "S13": "A08", "S14": "A06", "S15": "A05", "S16": "A05"}

repetition = ["N01", "N02"]

annotator_S01 = ["A17", "A12"]

labels_persons = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S11": 4, "S12": 5, "S13": 6, "S14": 7}
#in general from fernandos calculation
'''
NORM_MAX_THRESHOLDS = [392.85,    345.05,    311.295,    460.544,   465.25,    474.5,     392.85,
                       345.05,    311.295,   574.258,   575.08,    589.5,     395.81,    503.798,
                       405.9174,  322.9,     331.81,    338.4,     551.829,   598.326,   490.63,
                       667.5,     673.4,     768.6,     560.07,    324.22,    379.405,   193.69,
                       203.65,    159.297,   474.144,   402.57,    466.863,   828.46,    908.81,
                       99.14,    482.53,    381.34,    386.894,   478.4503,  471.1,     506.8,
                       420.04,    331.56,    406.694,   504.6,     567.22,    269.432,   474.144,
                       402.57,    466.863,   796.426,   863.86,    254.2,     588.38,    464.34,
                       684.77,    804.3,     816.4,     997.4,     588.38,    464.34,    684.77,
                       889.5,     910.6,    1079.7,     392.0247,  448.56,    673.49,    322.9,
                       331.81,    338.4,     528.83,    475.37,    473.09,    679.69,    735.2,
                       767.5,     377.568,   357.569,   350.501,   198.86,    197.66,    114.931,
                       527.08,    412.28,    638.503,   691.08,    666.66,    300.48,    532.11,
                       426.02,    423.84,    467.55,    497.1,     511.9,     424.76,    348.38,
                       396.192,   543.694,   525.3,     440.25,    527.08,    412.28,    638.503,
                       729.995,   612.41,    300.33,    535.94,    516.121,   625.628,   836.13,
                       920.7,     996.8,     535.94,    516.121,   625.628,   916.15,   1009.5,
                       1095.6,    443.305,   301.328,   272.984,   138.75,    151.84,    111.35]

NORM_MIN_THRESHOLDS = [-382.62, -363.81, -315.691, -472.2, -471.4, -152.398,
                       -382.62, -363.81, -315.691, -586.3, -581.46, -213.082,
                       -400.4931, -468.4, -409.871, -336.8, -336.2, -104.739,
                       -404.083, -506.99, -490.27, -643.29, -709.84, -519.774,
                       -463.02, -315.637, -405.5037, -200.59, -196.846, -203.74,
                       -377.15, -423.992, -337.331, -817.74, -739.91, -1089.284,
                       -310.29, -424.74, -383.529, -465.34, -481.5, -218.357,
                       -442.215, -348.157, -295.41, -541.82, -494.74, -644.24,
                       -377.15, -423.992, -337.331, -766.42, -619.98, -1181.528,
                       -521.9, -581.145, -550.187, -860.24, -882.35, -645.613,
                       -521.9, -581.145, -550.187, -936.12, -982.14, -719.986,
                       -606.395, -471.892, -484.5629, -336.8, -336.2, -104.739,
                       -406.6129, -502.94, -481.81, -669.58, -703.12, -508.703,
                       -490.22, -322.88, -322.929, -203.25, -203.721, -201.102,
                       -420.154, -466.13, -450.62, -779.69, -824.456, -1081.284,
                       -341.5005, -396.88, -450.036, -486.2, -486.1, -222.305,
                       -444.08, -353.589, -380.33, -516.3, -503.152, -640.27,
                       -420.154, -466.13, -450.62, -774.03, -798.599, -1178.882,
                       -417.297, -495.1, -565.544, -906.02, -901.77, -731.921,
                       -417.297, -495.1, -565.544, -990.83, -991.36, -803.9,
                       -351.1281, -290.558, -269.311, -159.9403, -153.482, -162.718]
'''
#type1 - avoiding subject 12

NORM_MAX_THRESHOLDS = [ 393.989,    287.475,    284.478,    460.544,    455.63,     460.81,
                       393.989,    287.475,    284.478,    574.258,    567.25,     575.75,
                       384.232,    425.469,    428.713,    319.7402,   328.464,    332.26,
                       501.277,    459.287,    420.799,    637.48,     701.72,     694.8,
                       369.9427,   282.033,    282.213,    189.599,    161.16,      47.46,
                       405.258,    279.757,    251.199,    823.932,    607.36,     -10.664,
                       444.912,    331.336,    386.894,   478.4503,   478.6046,   483.48,
                       415.752,    275.466,    388.655,    447.051,    495.98,    -30.379,
                       405.258,    279.757,    251.199,    742.349,    599.79,      75.128,
                       487.893,    430.53,     465.653,    759.099,    884.944,   912.23,
                       487.893,    430.53,     465.653,    857.174,    827.842,   1000.68,
                       359.8083,   415.3532,   391.604,    319.7402,   328.464,    332.26,
                       489.28,     471.109,    472.475,    671.4205,   652.6065,   730.2,
                       367.7923,   272.609,    280.378,    162.04,     183.507,     36.048,
                       404.989,    364.7847,   332.083,    430.64,     463.78,    -191.337,
                       526.335,    411.3173,   386.159,    457.298,    463.969,    489.08,
                       413.514,    331.942,    258.849,    421.433,    469.51,     -67.274,
                       404.989,   364.7847,   332.083,    617.79,     566.29,    -304.4075,
                       455.799,    438.55237,  625.628,    801.77,     816.391,    902.83,
                       455.799,    438.55237,  625.628,    901.27,     912.576,    998.04,
                       369.0953,   339.916,    265.986,    121.915,    117.66,      -2.37]

NORM_MIN_THRESHOLDS = [ -356.7484,  -350.385,   -275.538,   -426.97,    -459.003,   -133.613,
                       -356.7484,  -350.385,   -275.538,   -527.83,    -568.538,    -83.287,
                       -378.2979,  -389.603,   -409.797,   -307.85,    -318.67,     -90.61,
                       -333.4918,  -474.984,   -487.382,   -617.9,     -577.257,   -519.774,
                       -357.9123,  -259.305,   -280.37,    -181.0836,  -152.012,   -203.592,
                       -399.272,   -408.047,   -337.331,   -574.295,   -502.862,  -1044.2609,
                       -219.1371,  -349.727,   -386.167,   -435.15,    -457.899,   -218.357,
                       -445.881,   -300.465,   -253.822,   -433.78,    -442.28,    -628.13,
                       -399.272,   -408.047,   -337.331,   -573.84,   -566.73,   -1136.439,
                       -404.4907,  -470.094,   -466.894,   -843.88,    -721.09,    -675.655,
                       -404.4907,  -470.094,   -466.894,   -801.57,    -813.856,   -764.194,
                       -380.9074,  -372.724,   -324.768,   -307.85,    -318.67,     -90.61,
                       -392.5085,  -402.845,   -465.329,   -587.1,     -570.197,   -508.703,
                       -467.209,   -303.998,   -322.929,   -179.7976,  -164.472,   -185.146,
                       -292.192,   -323.723,   -282.578,   -726.54,    -604.11,   -1043.0801,
                       -376.8946,  -324.283,   -377.077,   -404.99,    -481.38,    -201.676,
                       -440.022,   -317.407,   -295.044,   -371.35,    -471.11,    -606.896,
                       -292.192,   -323.723,   -282.578,   -677.42,    -643.62,  -1094.1443,
                       -405.712,   -457.251,   -565.544,   -796.613,   -767.2,     -692.469,
                       -405.712,   -457.251,  -565.544,   -887.707,   -815.67,   -712.609,
                       -260.2279,   -98.0932,   -66.7167,  -122.788,   -107.3,     -162.189]

#type2 - avoiding subject 11
'''
NORM_MAX_THRESHOLDS = [ 385.977,   315.51,    284.478,   460.544,   455.554,   460.81,    385.977,
                       315.51,    284.478,   574.258,   545.438,   575.79,    407.0898,  361.089,
                       438.49,    319.7402,  328.464,   332.26,    551.829,   458.856,   456.418,
                       596.405,  701.72,    692.22,    368.975,   299.794,   295.802,   189.599,
                       168.743,    56.37,    474.144,   339.425,   466.863,   635.09,    674.39,
                       -337.071,   415.274,   331.336,   360.262,   478.4503,  478.6046,  484.78,
                       412.538,   318.036,   356.971,   443.76,    495.98,     11.23,    474.144,
                       339.425,   466.863,   557.87,    659.446,  -317.782,   553.026,   431.549,
                       454.361,   759.099,   884.944,   908.1,     553.026,   431.549,   454.361,
                       857.174,   935.67,    995.61,    371.714,   464.216,   398.4,     319.7402,
                       328.464,   332.26,    431.939,   470.874,   474.88,    679.3813,  652.6065,
                       730.2,     353.5983,  275.834,   290.669,   169.37,    168.33,     41.575,
                       410.511,   364.7847,  332.083,   691.08,    500.649,  -299.961,   451.959,
                       406.948,   397.648,   458.1778,  463.969,   490.44,    394.652,   314.628,
                       276.6189,  512.28,    469.51,    -51.053,  410.511,   364.7847,  332.083,
                       636.29,    567.933,  -379.0069,  377.973,   480.123,   465.931,   881.4743,
                       868.63,    903.52,    377.973,   480.123,   465.931,   959.3516,  948.31,
                       993.97,    105.2991,  173.9998,   73.915,   121.915,   119.16,    -14.362]

NORM_MIN_THRESHOLDS = [ -380.281,    -350.385,    -289.473,   -451.56,     -413.9324,   -133.818,
                       -380.281,    -350.385,    -289.473,    -555.71,     -522.335,    -124.513,
                       -400.7195,   -459.467,    -429.933,    -313.51,     -284.486,     -91.97,
                       -336.5175,   -483.53,     -480.473,    -662.14,     -538.202,   -424.506,
                       -342.8158,   -213.035,    -215.1083,   -163.4765,   -189.2,      -181.562,
                       -374.847,    -394.53,     -343.996,    -748.77,     -575.42,    -1055.5265,
                       -307.787,    -362.824,    -386.167,    -456.32,     -395.75682,  -169.694,
                       -316.4238,   -218.6021,   -211.421,    -460.37,     -403.06,     -616.43,
                       -374.847,    -394.53,     -343.996,    -732.32,     -524.04,    -1137.23,
                       -369.0439,   -482.376,    -466.985,    -896.76,     -723.832,   -620.108,
                       -369.0439,   -482.376,    -466.985,    -979.28,     -811.635,    -692.357,
                       -397.3337,   -413.817,    -358.3993,   -313.51,     -284.486,     -91.97,
                       -406.6129,   -444.227,    -452.954,    -647.92,     -641.364,    -490.251,
                       -392.95,     -287.97,     -294.898,    -201.08,     -177.62,     -173.816,
                       -347.006,    -463.64,     -390.582,    -779.69,     -647.2738,  -1051.8924,
                       -332.2788,   -310.893,    -377.077,    -408.43,     -386.25,     -222.305,
                       -400.798,    -319.012,    -279.312,    -516.3,      -464.34,     -608.365,
                       -347.006,    -463.64,     -390.582,    -774.03,     -625.93,    -1128.3,
                       -405.712,    -484.677,    -483.109,    -892.97,     -852.581,    -695.073,
                       -405.712,    -484.677,    -483.109,   -968.66,     -947.151,    -753.931,
                       -114.9842,   -132.7648,   -164.68537,  -131.22,     -117.78,     -162.387]

'''
#type3 - avoiding subjects 11 and 12
'''
NORM_MAX_THRESHOLDS = [ 385.977,   287.475,   284.478,   460.544,   455.63,    460.81,    385.977,
                       287.475,   284.478,   574.258,   567.25,    575.79,    407.0898,  425.469,
                       438.49,    319.7402,  328.464,   332.26,    501.277,   458.856,   441.221,
                       617.6725,  701.72,    694.8,     364.1056,  282.033,   282.213,   189.599,
                       161.16,     56.37,    474.144,   339.425,   466.863,   823.932,   632.929,
                       -10.664,   444.912,   331.336,   386.894,   478.4503,  478.6046,  483.48,
                       397.321,   275.466,   388.655,   447.051,   495.98,     11.23,    474.144,
                       339.425,   466.863,   742.349,   659.446,    75.128,   487.893,   431.549,
                       465.653,   759.099,   884.944,   912.23,    487.893,   431.549,   465.653,
                       857.174,   845.24,   1000.68,    371.714,   464.216,   398.4,     319.7402,
                       328.464,   332.26,    489.28,   471.109,   474.88,    679.3813,  652.6065,
                       730.2,     367.7923,  272.609,   278.983,   162.04,   183.507,    36.048,
                       394.282,   364.7847,  332.083,   480.337,   463.78,   -191.337,  526.335,
                       396.845,   397.648,   458.1778,  463.969,   490.44,    413.514,   320.786,
                       258.849,   512.28,    469.51,    -51.053,   394.282,   364.7847,  332.083,
                       592.81,    566.29,   -304.4075,  455.799,   480.123,   625.628,   881.4743,
                       816.391,   903.52,    455.799,   480.123,   625.628,   959.3516,  912.576,
                       998.04,    105.2991,  173.9998,  82.5642,  121.915,   117.66,     -2.37]

NORM_MIN_THRESHOLDS = [ -356.7484,   -350.385,    -287.806,   -451.56,     -459.003,     -97.146,
                       -356.7484,   -350.385,    -287.806,    -555.71,     -568.538,     -83.287,
                       -378.2979,   -459.467,    -409.797,    -313.51,     -318.67,      -68.252,
                       -336.5175,   -474.984,    -487.382,    -617.9,      -577.257,    -519.774,
                       -357.9123,   -259.305,    -262.553,    -181.0836,   -159.52,     -203.592,
                       -399.272,   -326.227,    -337.331,    -574.295,    -575.42,    -1055.5265,
                       -219.1371,   -349.727,    -386.167,    -456.32,     -457.899,    -218.357,
                       -445.881,    -298.987,    -236.365,    -433.78,     -403.06,     -628.13,
                       -399.272,    -326.227,   -337.331,    -573.84,     -566.73,    -1136.439,
                       -404.4907,   -470.094,   -466.894,    -843.88,    -715.859,    -645.613,
                       -404.4907,   -470.094,    -466.894,    -801.57,    -813.856,    -719.986,
                       -380.9074,   -372.724,    -351.265,    -313.51,     -318.67,      -68.252,
                       -392.5085,   -444.227,    -465.329,    -587.1,      -641.364,    -508.703,
                       -467.209,    -303.998,    -322.929,    -179.7976,   -177.62,    -185.146,
                       -292.192,    -463.64,     -390.582,    -726.54,     -604.11,    -1043.0801,
                       -308.7981,   -313.982,    -377.077,    -404.99,     -481.38,     -192.255,
                       -440.022,    -317.407,    -295.044,   -371.35,     -471.11,     -606.896,
                       -292.192,    -463.64,     -390.582,    -677.42,     -643.62,    -1095.6156,
                       -405.712,    -407.182,    -565.544,    -796.613,    -852.581,    -695.073,
                       -405.712,    -407.182,    -565.544,    -887.707,    -947.151,    -753.931,
                       -78.9523,    -98.0932,   -164.68537,  -131.22,     -117.78,     -162.387]

'''
#type4 - avoiding subjects 10,11 and 12
'''
NORM_MAX_THRESHOLDS = [ 385.977,    315.51,     284.478,    460.544,    455.63,     460.55,
                        385.977,    315.51,     284.478,    574.258,    567.25,     575.14,
                        407.0898,   425.469,    438.49,    319.7402,   316.33,     320.13,
                        501.277,    458.856,    441.221,    617.6725,   617.094,    694.8,
                        387.987,    297.434,    299.959,    189.599,    168.743,     56.37,
                        474.144,    339.425,    466.863,    823.932,    674.39,     -10.664,
                        444.912,    331.336,    386.894,    478.4503,   460.95,     478.41,
                        397.321,    288.675,    388.655,    447.051,    495.98,      11.23,
                        474.144,    339.425,    466.863,    742.349,    659.446,    75.128,
                        487.893,    431.549,    427.764,    759.099,    815.573,    903.74,
                        487.893,    431.549,    427.764,   857.174,    902.501,   990.93,
                        371.714,    464.216,    398.918,    319.7402,   316.33,     320.13,
                        489.28,     470.379,    472.247,    679.3813,   605.07,     730.2,
                        367.7923,   292.878,    295.373,    169.37,     183.507,     51.58,
                        394.282,    370.209,    332.083,    691.08,     500.649,   -191.337,
                        526.335,    420.442,    400.781,   457.298,    430,      483.15,
                        424.76,     320.786,    319.064,    512.28,     469.51,    -51.053,
                        394.282,    370.209,    332.083,    655.35,     566.29,    -304.4075,
                        455.799,    438.55237,  625.628,    881.4743,   772.9,     903.52,
                        455.799,    438.55237, 625.628,    959.3516,   826.32,     991.96,
                        369.03,     312.198,    257.93,     121.915,    117.66,      -2.37]

NORM_MIN_THRESHOLDS = [ -380.281,    -350.385,    -289.473,    -451.56,     -459.003,     -97.146,
                       -380.281,    -350.385,    -289.473,    -555.71,     -568.538,     -83.287,
                       -400.7195,   -459.467,    -429.933,    -313.51,     -318.67,      -68.252,
                       -336.5175,   -483.53,     -487.382,    -604.28,     -577.257,    -519.774,
                       -365.6905,   -260.096,    -276.414,    -194.38,     -189.2,      -203.592,
                       -399.272,    -394.53,     -343.996,    -739.58,     -575.42,    -1046.576,
                       -228.763,    -362.824,    -386.167,    -456.32,     -457.899,    -218.357,
                       -445.881,    -298.987,    -236.365,    -437.84,     -403.06,     -628.13,
                       -399.272,    -394.53,     -343.996,    -625.61,     -541.678,   -1137.23,
                       -404.4907,   -482.376,    -466.985,    -789.8,      -722.248,    -645.613,
                       -404.4907,   -482.376,    -466.985,    -865.37,     -813.856,    -719.986,
                       -380.9074,   -397.462,    -358.3993,   -313.51,     -318.67,      -68.252,
                       -406.6129,   -444.227,    -465.329,    -669.58,     -641.364,    -508.703,
                       -467.209,    -303.998,    -322.929,    -201.08,     -177.62,     -185.146,
                       -359.948,    -463.64,     -390.582,    -779.69,     -647.2738,  -1033.13373,
                       -341.5005,   -313.982,    -377.077,    -446.55,     -481.38,    -192.255,
                       -440.022,    -317.407,   -295.044,    -516.3,      -471.11,     -606.896,
                       -359.948,    -463.64,     -390.582,    -774.03,     -643.62,    -1128.3,
                       -405.712,    -478.784,    -565.544,    -906.02,     -852.581,    -692.469,
                       -405.712,    -478.784,    -565.544,    -990.83,     -947.151,    -703.37,
                       -351.1281,   -290.558,    -269.311,    -131.22,     -117.78,     -162.545]
'''

#def opp_sliding_window(data_x, data_y, ws, ss, label_pos_end=True):
def opp_sliding_window(data_x, data_y, ws, ss, label_pos_end=True):
    '''
    print("check1")
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    print(data_x.shape)
    
    #return data_x.astype(np.float32), data_y.astype(np.uint8)
    return data_x.astype(np.float32)
    '''
    print("Sliding window: Creating windows {} with step {}".format(ws, ss))

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    print(data_x.shape)
    # Label from the end
    if label_pos_end:
        
        data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
    else:
        if False:
            # Label from the middle
            # not used in experiments
           
            data_y_labels = np.asarray(
                [[i[i.shape[0] // 2]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
        else:
            # Label according to mode
            try:
                
                data_y_labels = []
                for sw in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1)):
                   
                    labels = np.zeros((1)).astype(int)
                    count_l = np.bincount(sw[:, 0], minlength=NUM_CLASSES)
                    idy = np.argmax(count_l)
                    labels[0] = idy
                   
                    data_y_labels.append(labels)
                data_y_labels = np.asarray(data_y_labels)


            except:
                print("Sliding window: error with the counting {}".format(count_l))
                print("Sliding window: error with the counting {}".format(idy))
                return np.Inf

            # All labels per window
            data_y_all = np.asarray([i[:,0] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
            print(data_y_all.shape)
            
    print("daya_y_labels")
    print(data_y_labels.shape)
    print("daya_y_all")
    print(data_y_all.shape)

    return data_x.astype(np.float32), data_y_labels.astype(np.uint8), data_y_all.astype(np.uint8)


def normalize(data):
    """
    Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    try:
        max_list, min_list = np.array(NORM_MAX_THRESHOLDS), np.array(NORM_MIN_THRESHOLDS)
        diffs = max_list - min_list
        for i in np.arange(data.shape[1]):
            data[:, i] = (data[:, i]-min_list[i])/diffs[i]
        #     Checking the boundaries
        data[data > 1] = 0.99
        data[data < 0] = 0.00
    except:
        raise("Error in normalization")
        
    return data

def divide_x_y(data):
    """Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """
    data_t = data[:, 0]
    data_y = data[:, 1]
    data_x = data[:, 2:]
    
    return data_t, data_x, data_y

def select_columns_opp(data):
    """
    Selection of the columns employed in the MoCAP
    excluding the measurements from lower back,
    as this became the center of the human body,
    and the rest of joints are normalized
    with respect to this one

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: numpy integer matrix
        Selection of features
    """

    #included-excluded
    features_delete = np.arange(68, 74)
    
    return np.delete(data, features_delete, 1)

def generate_data(ids, sliding_window_length, sliding_window_step, data_dir=None, identity_bool=False, usage_modus='train'):
     #type1-avoiding person 12
    
    persons = ["S07", "S08", "S09", "S10", "S11", "S13", "S14"]
    ID = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S11": 4, "S13": 5, "S14": 6}
    train_ids = ["R03", "R07", "R08", "R10", "R11"]
    val_ids = ["R12"]
    test_ids = ["R15"]
    
    #type2- avoiding person 11
    '''
    persons = ["S07", "S08", "S09", "S10", "S12", "S13", "S14"]
    ID = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S12": 4, "S13": 5, "S14": 6}
    train_ids =["R11", "R12", "R15", "R18", "R19","R21"]
    val_ids = ["R22"]
    test_ids = ["R23"]
    
    '''
    #type3- Avoiding person 11 and 12
    '''
    persons = ["S07", "S08", "S09", "S10", "S13", "S14"]
    ID = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S13": 4, "S14": 5}
    train_ids = ["R03", "R07", "R08", "R10", "R11", "R12", "R15", "R18"]
    val_ids = ["R19", "R21"]
    test_ids = ["R22", "R23"]
    '''
    
    #type4-Avoiding persons 11,12,10
    '''
    persons = ["S07", "S08", "S09", "S13", "S14"]
    ID = {"S07": 0, "S08": 1, "S09": 2, "S13": 3, "S14": 4}
    train_ids = ["R03", "R07", "R08", "R10", "R11", "R12", "R15", "R18", "R19", "R21", "R22"]
    val_ids = ["R23","R25", "R26"]
    test_ids = ["R27", "R28", "R29"]
    '''
    counter_seq = 0
    
    for P in persons:
        if usage_modus == 'train':
           recordings = train_ids
        elif usage_modus == 'val':
            recordings = val_ids
        elif usage_modus == 'test':
            recordings = test_ids
        print("\nModus {} \n{}".format(usage_modus, recordings))
        for R in recordings:
            S = SCENARIO[R]
            for N in repetition:
                    annotator_file = annotator[P]
                    if P == 'S07' and SCENARIO[R] == 'L01':
                        annotator_file = "A03"
                    if P == 'S14' and SCENARIO[R] == 'L03':
                        annotator_file = "A19"
                    if P == 'S11' and SCENARIO[R] == 'L01':
                        annotator_file = "A03"
                    if P == 'S11' and R in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                        annotator_file = "A02"
                    if P == 'S13' and R in ['R28']:
                        annotator_file = "A01"
                    if P == 'S13' and R in ['R29', 'R30']:
                        annotator_file = "A11"
                    if P == 'S09' and R in ['R28', 'R29']:
                        annotator_file = "A01"
                    if P == 'S09' and R in ['R21', 'R22', 'R23', 'R24', 'R25']:
                        annotator_file = "A11"
                        
                    file_name_norm = "{}/{}_{}_{}_{}_{}_norm_data.csv".format(P, S, P, R, annotator_file, N)
                    file_name_label = "{}/{}_{}_{}_{}_{}_labels.csv".format(P, S, P, R, annotator_file, N)
                    try:
                        data = csv_reader.reader_data(FOLDER_PATH + file_name_norm)
                        print("\nFiles loaded in modus {}\n{}".format(usage_modus, file_name_norm))
                        data = select_columns_opp(data)
                        print("\nFiles loaded")
                    except:
                        print("\n1 In loading data,  in file {}".format(FOLDER_PATH + file_name_norm))
                        continue
                    
                    try:
                        #Getting labels and attributes
                        labels = csv_reader.reader_labels(FOLDER_PATH + file_name_label)
                        class_labels = np.where(labels[:, 0] == 7)[0]

                        # Deleting rows containing the "none" class
                        data = np.delete(data, class_labels, 0)
                        labels = np.delete(labels, class_labels, 0)
                        act_class= labels
                        
                        downsampling = range(0, data.shape[0], 2)
                        data = data[downsampling]
                        act_class = act_class[downsampling]
                    except:
                        print("\n In generating data, Error getting the data {}".format(FOLDER_PATH + file_name_norm))
                        continue
                    
                    labelid=ID[P]
                    print("label")
                    print(labelid)
                    data_t, data_x, data_y = divide_x_y(data)
                    
                    del data_t
                    #    print("\n In generating data, Error getting the data {}".format(FOLDER_PATH + file_name_norm))
                    #    continue
                    try:
                        # checking if annotations are consistent
                        data_x = normalize(data_x)
                        print("data shape")
                        print(data_x.shape)
                                                
                        if data_x.shape[0] == data_x.shape[0]:
                            print("Starting sliding window")
                            X, y, y_all = opp_sliding_window(data_x, act_class.astype(int), sliding_window_length, sliding_window_step, label_pos_end = False)
                            print("Windows are extracted")
                                                        
                            for f in range(X.shape[0]):
                                try:

                                    sys.stdout.write('\r' + 'Creating sequence file '
                                                            'number {} with id {}'.format(f, counter_seq))
                                    sys.stdout.flush()

                                    # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                                    seq = np.reshape(X[f], newshape = (1, X.shape[1], X.shape[2]))
                                    seq = np.require(seq, dtype=np.float)
                                    
                                    # Storing the sequences
                                    #obj = {"data": seq, "label": labelid}
                                    obj = {"data": seq, "act_label": y[f], "act_labels_all": y_all[f], "label": labelid}
                                    f = open(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                                    f.close()

                                    counter_seq += 1
                                except:
                                    raise('\nError adding the seq')

                            print("\nCorrect data extraction from {}".format(FOLDER_PATH + file_name_norm))

                            del data
                            del data_x
                            del data_y
                            del X
                            del labels
                            del class_labels

                        else:
                            print("\nNot consisting annotation in  {}".format(file_name_norm))
                            continue

                    except:
                        print("\n In generating data, No file {}".format(FOLDER_PATH + file_name_norm))
            
    return

def generate_CSV(csv_dir, data_dir):
    '''
    Generate CSV file with path to all (Training) of the segmented sequences
    This is done for the DATALoader for Torch, using a CSV file with all the paths from the extracted
    sequences.

    @param csv_dir: Path to the dataset
    @param data_dir: Path of the training data
    '''
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for n in range(len(filenames)):
            f.append(data_dir + 'seq_{0:06}.pkl'.format(n))

    #np.savetxt(csv_dir + type_file, f, delimiter="\n", fmt='%s')
    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')
    
    return

def generate_CSV_final(csv_dir, data_dir1, data_dir2):
    '''
    Generate CSV file with path to all (Training and Validation) of the segmented sequences
    This is done for the DATALoader for Torch, using a CSV file with all the paths from the extracted
    sequences.

    @param csv_dir: Path to the dataset
    @param data_dir1: Path of the training data
    @param data_dir2: Path of the validation data
    '''
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir1):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:06}.pkl'.format(n))

    for dirpath, dirnames, filenames in os.walk(data_dir2):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:06}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')

    return
                            
                            
def create_dataset():
    #type1-avoiding person 12
    
    train_ids = ["R03", "R07", "R08", "R10", "R11"]
    val_ids = ["R12"]
    test_ids = ["R15"]
   
    #type2- avoiding person 11
    '''
    train_ids = ["R11", "R12", "R15", "R18", "R19", "R21"]
    val_ids = ["R22"]
    test_ids = ["R23"]
    
    '''
    #type3- Avoiding person 11 and 12
    '''
    train_ids = ["R03", "R07", "R08", "R10" "R11", "R12", "R15", "R18"]
    val_ids = ["R19", "R21"]
    test_ids = ["R22", "R23"]
    '''
    
    #type4-Avoiding persons 11,12,10
    '''
    train_ids = ["R03", "R07", "R08", "R10" "R11", "R12", "R15", "R18", "R19", "R21", "R22"]
    val_ids = ["R23","R25", "R26"]
    test_ids = ["R27", "R28", "R29"]
    '''
    
    base_directory = '/data/nnair/output/activities/type1/mocap/'
    sliding_window_length = 100
    sliding_window_step = 12
    
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    
    generate_data(train_ids, sliding_window_length=sliding_window_length,
                  sliding_window_step=sliding_window_step, data_dir=data_dir_train, usage_modus='train')
    generate_data(val_ids, sliding_window_length=sliding_window_length,
                  sliding_window_step=sliding_window_step, data_dir=data_dir_val, usage_modus='val')
    generate_data(test_ids, sliding_window_length=sliding_window_length,
                  sliding_window_step=sliding_window_step, data_dir=data_dir_test, usage_modus='test')
    
    generate_CSV(base_directory + "train.csv", data_dir_train)
    generate_CSV(base_directory + "val.csv", data_dir_val)
    generate_CSV(base_directory + "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)
    
    return


#######################

if __name__ == '__main__':
    
    create_dataset()

    print("Done")