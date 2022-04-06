'''
Created on Oct 22, 2021

@author: fmoya

HAR using Pytorch
Code updated from caffe/theano implementations
'''

import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import datetime
import scipy.interpolate
from scipy.spatial import distance

import xml.etree.ElementTree as ET

import pympi as pympi
import pickle

from sliding_window import sliding_window

classes_labels = {"clean": 0, "close": 1, "cut": 2, "fill": 3, "open": 4, "other": 5, "put": 6,
                    "shake": 7, "stir": 8, "take": 9, "turn_off": 10, "turn_on": 11, "walk": 12}



def read_attr_rep(pathfile):

    with open(pathfile, 'r') as csvfile:
        print(pathfile)
        totalactivities = []
        activities = []
        attrs_rep = []
        attr = csv.reader(csvfile, delimiter='\n', quotechar='|')
        for row in attr:
            try:
                if attr.line_num == 1:
                    print(', '.join(row))
                else:
                    splitrow = row[0].split("\t")
                    totalactivities.append(splitrow[1])
                    activities.append(splitrow[2])
                    attr_rep = []
                    for singleattr in splitrow[3:]:
                        attr_rep.append(int(singleattr))
                    attr_rep = np.array(attr_rep)
                    attrs_rep.append(attr_rep)
                    #print(', '.join(row))
            except KeyboardInterrupt:
                print('\nYou cancelled the operation.')

    return totalactivities, activities, np.array(attrs_rep)

def totalmillis(t, datetimeBool=False):
    if datetimeBool:
        tm = (t.second * 1000) + (t.microsecond / 1000)
    else:
        tm = (t.seconds * 1000) + (t.microseconds / 1000)

    return tm

def read_3DMGX1(pathfile):

    sensor_IDs = ["2794", "2795", "2796", "3261", "3337"]
    DMGX1_recordings = {}

    for root_d, dirs_D, files in os.walk(pathfile):
        for fi in files:
        #    fi_split = fi.split('_')[0]
        #    data = np.loadtxt(pathfile + "/" + fi, delimiter=',', skiprows=2)
            with open(pathfile + fi, 'r') as csvfile:
                print(fi)
                data = []
                time_r = []
                DMGX1 = csv.reader(csvfile, delimiter='\n', quotechar='|')
                for row in DMGX1:
                    try:
                        if DMGX1.line_num < 3:
                            print(', '.join(row))
                        else:
                            splitrow = row[0].split("\t")
                            data.append(list(map(float, splitrow[:-1])))
                            #if 723 <= DMGX1.line_num < 732:
                            #    print(row)
                            #print(', '.join(row))
                            if int(splitrow[-1].split("_")[-1]) % 10:
                                milli_corrected = (int(splitrow[-1].split("_")[-1]) - 1)//10 #(int(splitrow[-1].split("_")[-1]) % 10)
                                if milli_corrected == 0:
                                    strmillis = splitrow[-1].split("_")[0] + '_' +  splitrow[-1].split("_")[1] + '_' + splitrow[-1].split("_")[2] + '_0000000'
                                else:
                                    strmillis = splitrow[-1].split("_")[0] + '_' +  splitrow[-1].split("_")[1] + '_' + splitrow[-1].split("_")[2] + '_' + '{:06d}'.format(milli_corrected)

                                time_r.append(datetime.datetime.strptime(strmillis, '%H_%M_%S_%f0'))
                            else:
                                time_r.append(datetime.datetime.strptime(splitrow[-1], '%H_%M_%S_%f0'))

                    except KeyboardInterrupt:
                        print('\nYou cancelled the operation.')

                DMGX1_recordings[fi] = [np.array(data), time_r]

                """
                data = np.array(data)
                plt.plot(data[:, -1], data[:, 0], 'r')
                plt.legend([fi])
                #plt.axis([-0.05, 6.33, -1.05, 1.05])
                plt.title(fi)
                #plt.show()
                plt.pause(5.0)
                plt.close()
                """

    return DMGX1_recordings


def read_6DOFv4(pathfile):

    sensor_IDs = ["000666029DE1", "0006660160E3", "000666015711", "000666015735"]

    DOFv4_recordings = {}
    flag_time_error = False
    for root_d, dirs_D, files in os.walk(pathfile):
        for fi in files:
            #    fi_split = fi.split('_')[0]
            #    data = np.loadtxt(pathfile + "/" + fi, delimiter=',', skiprows=2)
            with open(pathfile + fi, 'r') as csvfile:
                print(fi)
                data = []
                time_r = []
                DOFv4 = csv.reader(csvfile, delimiter='\n', quotechar='|')
                for row in DOFv4:
                    try:
                        if DOFv4.line_num < 3:
                            print(', '.join(row))
                        else:
                            #if 15370 < DOFv4.line_num < 15395:
                            #    print(row)
                            #print(', '.join(row))
                            splitrow = row[0].split("\t")
                            if splitrow[-2].split("_")[0] == 'ERROR':
                                splitrow[-2] = -1
                                flag_time_error = True
                            data.append(list(map(float, splitrow[:-1])))

                            #if 2969 <= DOFv4.line_num < 2973 and fi == '000666015711_02-13_15_52_13-time.txt':
                            #    print(row)
                            if int(splitrow[-1].split("_")[-1]) % 10:
                                milli_corrected = (int(splitrow[-1].split("_")[-1]) - 1)//10 #(int(splitrow[-1].split("_")[-1]) % 10)
                                if milli_corrected == 0:
                                    strmillis = splitrow[-1].split("_")[0] + '_' +  splitrow[-1].split("_")[1] + '_' + splitrow[-1].split("_")[2] + '_0000000'
                                else:
                                    strmillis = splitrow[-1].split("_")[0] + '_' +  splitrow[-1].split("_")[1] + '_' + splitrow[-1].split("_")[2] + '_' + '{:06d}'.format(milli_corrected)

                                if milli_corrected % 10 > 0:
                                    time_r.append(datetime.datetime.strptime(strmillis, '%H_%M_%S_%f'))
                                else:
                                    time_r.append(datetime.datetime.strptime(strmillis, '%H_%M_%S_%f0'))
                            else:
                                time_r.append(datetime.datetime.strptime(splitrow[-1], '%H_%M_%S_%f0'))

                            #print(', '.join(row))
                    except KeyboardInterrupt:
                        print('\nYou cancelled the operation.')

                DOFv4_recordings[fi] = [np.array(data), time_r]

                """
                data = np.array(data)
                plt.plot(data[:, -1], data[:, 0], 'r')
                plt.legend([fi])
                #plt.axis([-0.05, 6.33, -1.05, 1.05])
                plt.title(fi)
                #plt.show()
                plt.pause(5.0)
                plt.close()
                """

    if flag_time_error:
        print("correcting sequence")
    return DOFv4_recordings


def read_eWatch(pathfile):

    sensor_IDs = ["AccX.txt", "AccY.txt", "AccZ.txt"]

    AccX = np.loadtxt(pathfile + sensor_IDs[0], delimiter=',', skiprows=0)
    AccY = np.loadtxt(pathfile + sensor_IDs[1], delimiter=',', skiprows=0)
    AccZ = np.loadtxt(pathfile + sensor_IDs[2], delimiter=',', skiprows=0)

    data = [AccX, AccY, AccZ]
    data = np.array(data).transpose()
    plt.plot(np.arange(0, data.shape[0]), data[:, 0], 'r')
    plt.legend(["ACC"])
    #plt.axis([-0.05, 6.33, -1.05, 1.05])
    plt.title("ACC")
    #plt.show()
    plt.pause(5.0)
    plt.close()

    return


def read_sync_video_time(pathfile):

    with open(pathfile, 'r') as csvfile:
        print(pathfile)
        sync_time_video = csv.reader(csvfile, delimiter='\n', quotechar='|')
        for row in sync_time_video:
            try:
                if sync_time_video.line_num == 1:
                    print(', '.join(row))
                    splitrow = row[0].split("EPOCH: -- ")
                    time_r = datetime.datetime.strptime(splitrow[-1], '%H_%M_%S_%f0')
                    break
            except KeyboardInterrupt:
                print('\nYou cancelled the operation.')

    return time_r


def read_annotations(pathfile):

    fileEAF = pympi.Eaf(pathfile)

    data_timeslots = fileEAF.timeslots
    data_tiers = fileEAF.tiers['default'][0]

    annotations = {}

    for ann, ann_data in data_tiers.items():
        annotations[ann] = (data_timeslots[ann_data[0]] - data_timeslots['ts1'],
                            data_timeslots[ann_data[1]] - data_timeslots['ts1'], ann_data[2])

    return annotations


def load_training(data_set="training"):
    pathfolder = "/data/fmoya/Doktorado/ARDUOUS_challenge/kitchen_dataset_tools/challenge-ml/cmu-data/" + data_set + "/"
    annotationsfolder = "/data/fmoya/Doktorado/ARDUOUS_challenge/kitchen_dataset_tools/challenge-ml/rostock-cmu-semantic-annotation/Plans/"

    if data_set == "training":
        subjects = {"S47": "Brownie", "S54": "Brownie", "S13": "Brownie", "S31": "Brownie",
                    "S12": "Sandwich", "S16": "Sandwich", "S25": "Sandwich", "S34": "Sandwich",
                    "S28": "Eggs", "S08": "Eggs", "S20": "Eggs", "S162": "Eggs"}

        subjects = {"S47": "Brownie", "S54": "Brownie", "S13": "Brownie", "S31": "Brownie",
                    "S12": "Sandwich", "S16": "Sandwich", "S25": "Sandwich", "S34": "Sandwich",
                    "S28": "Eggs", "S20": "Eggs", "S162": "Eggs"}



    elif data_set == "test":
        subjects = {"S09": "Brownie", "S15": "Sandwich", "S50": "Eggs"}

        subjects = {"S50": "Eggs"}
    #subjects = {"S08": "Eggs", "S20": "Eggs", "S162": "Eggs"}


    for subject, task in subjects.items():
        print("\n")
        print(subject, task)
        DMGX1 = read_3DMGX1(pathfolder + subject + "/" + subject + "_" + task + "_3DMGX1/")
        DOFv4 = read_6DOFv4(pathfolder + subject + "/" + subject + "_" + task + "_6DOFv4/")
        annotations = read_annotations(annotationsfolder + task + "/" + subject + ".eaf")
        sync_init = read_sync_video_time(pathfolder + subject + "/" + subject + "_" + task + "_Video/STime7150991-time-synch.txt")
        totalactivities, activities, attrs_rep = read_attr_rep("annotations_" + task + ".txt")
        #eWatch = read_eWatch(pathfolder + subject + "/" + subject + "_" + task + "_eWatch/")

        if True:
            data = DMGX1.copy()
            data.update(DOFv4)

            mintime = min([v[-1][0] for k, v in data.items()])
            maxtime = max([v[-1][-1] for k, v in data.items()])

            # Setting up the time in milliseconds
            data_seconds = {}
            for recordings, recording_data in data.items():
                recording_data_sec = np.zeros((recording_data[0].shape[0], recording_data[0].shape[1] + 1))
                time_r_sec = []
                for time_r in recording_data[-1]:
                    time_r_sec.append(totalmillis(time_r - mintime, datetimeBool=False))
                    #time_r_sec.append(totalmillis(time_r - recording_data[-1][0], datetimeBool=False))
                recording_data_sec[:, 0] = np.array(time_r_sec).transpose()
                recording_data_sec[:, 1] = recording_data[0][:, -1]
                recording_data_sec[:, 2:] = recording_data[0][:, :-1]
                data_seconds[recordings] = recording_data_sec

            endtime_annotations = annotations[list(annotations.keys())[-1]][1]

            mintime = min([v[-1][0] for k, v in data.items()])
            maxmintime = max([v[-1][0] for k, v in data.items()])
            minmaxtime = min([v[-1][-1] for k, v in data.items()])
            inittime = maxmintime - mintime
            endtime = minmaxtime - mintime
            inittime = totalmillis(inittime, datetimeBool=False)
            endtime = totalmillis(endtime, datetimeBool=False)
            if subject == 'S08':
                inittime_corrected = endtime - annotations[list(annotations.keys())[-1]][1]
            else:
                #inittime_corrected = inittime
                inittime_corrected = totalmillis(sync_init - mintime)
            #new_time = np.arange(inittime_corrected, endtime, 10)#[:200]
            new_time = np.arange(inittime_corrected, inittime_corrected + annotations[list(annotations.keys())[-1]][1], 5)#[:600]
            print(new_time.shape)

            #data = np.empty((new_time.shape[0], recording_data.shape[1]))

            recordings_interpolated = {}
            for recordings, recording_data in data_seconds.items():
                final_time = []
                print(recording_data.shape)
                #recording_data_sec = np.zeros((new_time.shape[0], recording_data.shape[1]))

                time_sampling = recording_data[:, 0]

                measurements = recording_data[:, 2:]
                print(measurements.shape)


                r1 = 0
                r2 = 0

                measurements_new = [np.empty((0)) for ss in range(measurements.shape[1])]
                print(measurements_new[0].shape)
                # time_new = [np.empty((0)) for ss in range(measurements.shape[1])]
                print("\nLength new time {} differences {}".format(len(new_time), len(time_sampling)))
                for tm in range(0, len(new_time) - 3, 3):
                    # look for time range
                    for tmr in range(len(time_sampling)):
                        if new_time[tm] <= time_sampling[tmr]:
                            r1 = tmr
                            break

                    for tmr in range(len(time_sampling)):
                        if new_time[tm + 3] <= time_sampling[tmr]:
                            r2 = tmr
                            break

                    r1 = r1 - 4
                    if r1 < 0:
                        r1 = 0

                    r2 = r2 + 3
                    if r2 >= len(time_sampling):
                        r2 = len(time_sampling) - 1

                    if r1 == 0:
                        r2 = r2 + 2

                    y = measurements[r1: r2]
                    x = time_sampling[r1: r2]

                    if new_time[tm] > time_sampling[-1] or new_time[tm] < time_sampling[0]:
                        print("Out of boundaries {}".format(tm))
                        print(new_time[tm - 5: tm + 5])
                        print("r1: {} tm: {} r2 {}:".format(r1, tm, r2))

                    sys.stdout.write('\r' + 'Sensor: ' + recordings + ' time :' + str(tm) + ' out of ' + str(len(new_time))
                                     + " r1:" + str(r1) + " r2: " + str(r2) + "   "
                                     + ' time :' + str(new_time[tm]) + " r1:" + str(time_sampling[r1]) +
                                     " r2: " + str(time_sampling[r2]) + "             ")
                    sys.stdout.flush()

                    #for dim in range(2, recording_data.shape(1)):
                        #measurements = recording_data[dim]

                    x_new = new_time[tm: tm + 3]
                    for ms in range(0, measurements.shape[1]):
                        # sys.stdout.write('\r' + 'x size ' + str(len(x)) + 'y size {}' + str(len(y[:, ms])))
                        # sys.stdout.flush()
                        try:
                            #print("\n")
                            #print(x)
                            #print(y[:, ms])
                            tck = scipy.interpolate.splrep(x, y[:, ms], s=0, k=3)
                            y_new = scipy.interpolate.splev(x_new, tck, der=0)

                            measurements_new[ms] = np.append(measurements_new[ms], y_new, axis=0)
                            # time_new[ms] = np.append(time_new[ms], x_new, axis=0)
                        except:
                            print('\nerror at Interpolation in  S:{} T: {} R: {}'. format(subject, task, recordings))
                    final_time.append(x_new)
                recordings_interpolated[recordings] = (np.array(measurements_new),
                                                       np.array(final_time).reshape((-1)))

            data = []
            for recordings, recording_data in recordings_interpolated.items():
                print("\n")
                print(recordings)
                print(recording_data[0].shape, recording_data[1].shape)
                data.append(recording_data[0])

            final_time = recording_data[1] - inittime_corrected
            data = np.array(data)
            print("data shape {}".format(data.shape))
            data = data.reshape((-1, data.shape[2])).transpose()

            """
            plt.plot(final_time, data[:, 0], 'r')
            plt.legend(["fi"])
            # plt.axis([-0.05, 6.33, -1.05, 1.05])
            plt.title("fi")
            # plt.show()
            plt.pause(5.0)
            plt.close()
            """


            all_annotations = []
            for ft in range(final_time.shape[0]):
                for ann, annlabel in annotations.items():
                    if annlabel[0] <= ft < annlabel[1]:
                        label = annlabel[2]
                        break
                all_annotations.append(annlabel[2])

            annotations_attr_rep = []
            annotation_activities = []
            for idxann, ann in enumerate(all_annotations):
                for idxallact, allact in enumerate(totalactivities):
                    if (ann == allact):
                        #print(idxallact)
                        annotations_attr_rep.append(attrs_rep[idxallact])
                        annotation_activities.append(activities[idxallact])
                        break

            annotations_attr_rep = np.array(annotations_attr_rep)

            obj = {"data": data, "labels": all_annotations, "activities": annotation_activities,
                   "attr_reps": annotations_attr_rep, "final_time": final_time}
            file_name = open(pathfolder + subject + "/" + subject + "_" + task + ".pkl", 'wb')
            pickle.dump(obj, file_name, protocol=pickle.HIGHEST_PROTOCOL)
            file_name.close()

            print("Created file :" + subject + "/" + subject + "_" + task + ".pkl")


    return



################
# Generate data
#################
def generate_data(ids, sliding_window_length, sliding_window_step, data_dir=None,
                  identity_bool = False, usage_modus = 'train'):
    '''
    creates files for each of the sequences, which are extracted from a file
    following a sliding window approach

    returns
    Sequences are stored in given path

    @param ids: ids for train, val or test
    @param sliding_window_length: length of window for segmentation
    @param sliding_window_step: step between windows for segmentation
    @param data_dir: path to dir where files will be stored
    @param identity_bool: selecting for identity experiment
    @param usage_modus: selecting Train, Val or testing
    '''

    FOLDER_PATH = "/data/fmoya/HAR/datasets/cmu-data/training_files/"
    #FOLDER_PATH = "/data/fmoya/Doktorado/ARDUOUS_challenge/kitchen_dataset_tools/challenge-ml/cmu-data/training/"

    number_of_attr = {"Brownie": [0, 88], "Eggs": [88, 165], "Sandwich": [165, 216]}

    counter_seq = 0
    hist_classes_all = np.zeros((len(classes_labels)))
    counter_file_label = -1

    for subject, task in ids.items():
        print("\n")
        print(subject, task)
        #file_name_data = subject + '/' + subject + "_" + task + ".pkl"
        file_name_data = subject + "_" + task + ".pkl"

        try:
            # getting data
            f = open(FOLDER_PATH + file_name_data, 'rb')
            data_all = pickle.load(f, encoding='bytes')
            f.close()

            totalactivities, activities, attrs_rep = read_attr_rep("annotations_" + task + ".txt")


            data_x = data_all["data"]
            labels = data_all["labels"]
            annotation_activities = data_all["activities"]
            annotations_attr_rep = data_all["attr_reps"]

            attr_rep_tasks = np.zeros((annotations_attr_rep.shape[0], 133 + 2))
            #attr_rep_tasks[:, number_of_attr[task][0] + 2: number_of_attr[task][1] + 2] = annotations_attr_rep
            attr_rep_tasks[:, 2:] = annotations_attr_rep

            annotation_activities_idx = []
            for activ in annotation_activities:
                annotation_activities_idx.append(classes_labels[activ])
            annotation_activities_idx = np.array(annotation_activities_idx)

            attr_rep_tasks[:, 1] = annotation_activities_idx

            #final_time = data_all["final_time"]

            attr_rep_tasks[:, 0] = data_all["final_time"]

            print("\nFiles loaded in modus {}\n{}".format(usage_modus, file_name_data))
            print("\nFiles loaded")
        except:
            print("\n1 In loading data,  in file {}".format(FOLDER_PATH + file_name_data))
            continue

        try:
            max_values, min_values, mean_values, std_values = statistics_measurements()
            data_x = norm_mbientlab(data_x, mean_values, std_values)
            print("\nData normalized")

        except:
            print("Error at normalizing")

        try:
            # checking if annotations are consistent
            if data_x.shape[0] == data_x.shape[0]:
                # Sliding window approach
                print("\nStarting sliding window")
                X, y, y_all = opp_sliding_window(data_x, attr_rep_tasks.astype(int),
                                                 sliding_window_length,
                                                 sliding_window_step, label_pos_end=False)
                print("\nWindows are extracted")

                # Statistics

                hist_classes = np.bincount(y[:, 0], minlength=len(classes_labels))
                hist_classes_all += hist_classes
                print("\nNumber of seq per class {}".format(hist_classes_all))

                counter_file_label += 1

                for f in range(X.shape[0]):
                    try:

                        # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                        seq = np.reshape(X[f], newshape=(1, X.shape[1], X.shape[2]))
                        seq = np.require(seq, dtype=np.float)

                        obj = {"data": seq, "label": y[f], "labels": y_all[f], "label_file": counter_file_label,
                               "task": task}
                        file_name = open(os.path.join(data_dir,
                                                      'seq_{0:07}.pkl'.format(counter_seq)), 'wb')
                        pickle.dump(obj, file_name, protocol=pickle.HIGHEST_PROTOCOL)

                        counter_seq += 1

                        sys.stdout.write(
                            '\r' +
                            'Creating sequence file number {} with id {}'.format(f, counter_seq))
                        sys.stdout.flush()

                        file_name.close()

                    except:
                        raise ('\nError adding the seq')

                print("\nCorrect data extraction from {}".format(FOLDER_PATH + file_name_data))

                del data_x
                del X
                del labels

            else:
                print("\n4 Not consisting annotation in  {}".format(file_name_data))
                continue
        except:
            print("\n5 In generating data, No created file {}".format(FOLDER_PATH + file_name_data))
        print("-----------------\n{}\n{}\n-----------------".format(subject, task))

    return


def statistics_measurements():
    '''
    Compute some statistics of the duration of the sequences data:

    print:
    Max and Min durations per class or attr
    Mean and Std durations per class or attr

    @param
    '''

    subjects = {"S47": "Brownie", "S54": "Brownie", "S13": "Brownie", "S31": "Brownie",
                "S12": "Sandwich", "S16": "Sandwich", "S25": "Sandwich", "S34": "Sandwich",
                "S28": "Eggs", "S20": "Eggs", "S162": "Eggs"}

    FOLDER_PATH = "/data/fmoya/HAR/datasets/cmu-data/training_files/"
    FOLDER_PATH = "/data/fmoya/Doktorado/ARDUOUS_challenge/kitchen_dataset_tools/challenge-ml/cmu-data/training/"

    accumulator_measurements = np.empty((0, 81))

    for subject, task in subjects.items():
        print("\n")
        print(subject, task)
        file_name_data = subject + '/' + subject + "_" + task + ".pkl"

        try:
            # getting data
            f = open(FOLDER_PATH + file_name_data, 'rb')
            data_all = pickle.load(f, encoding='bytes')
            f.close()

            print("------------------------------\n{}\n{}".format(subject, task))
            try:
                # getting data
                data_x = data_all["data"]

                accumulator_measurements = np.append(accumulator_measurements, data_x, axis = 0)
                print("\nFiles loaded")
            except:
                print("\n1 In loading data,  in file {}".format(FOLDER_PATH + file_name_data))
                continue
        except:
            print("\n1 In loading data,  in file {}".format(FOLDER_PATH + file_name_data))
            continue

    try:
        max_values = np.max(accumulator_measurements, axis=0)
        min_values = np.min(accumulator_measurements, axis=0)
        mean_values = np.mean(accumulator_measurements, axis=0)
        std_values = np.std(accumulator_measurements, axis=0)
    except:
        max_values = 0
        min_values = 0
        mean_values = 0
        std_values = 0
        print("Error computing statistics")

    return max_values, min_values, mean_values, std_values


def norm_mbientlab(data, mean_values, std_values):

    mean_values = np.reshape(mean_values, [1, 81])

    std_values = np.reshape(std_values, [1, 81])

    mean_array = np.repeat(mean_values, data.shape[0], axis=0)
    std_array = np.repeat(std_values, data.shape[0], axis=0)

    max_values = mean_array + 2 * std_array
    min_values = mean_array - 2 * std_array

    data_norm = (data - min_values) / (max_values - min_values)

    data_norm[data_norm > 1] = 1
    data_norm[data_norm < 0] = 0

    #data_norm = (data - mean_array) / std_array

    return data_norm


def opp_sliding_window(data_x, data_y, ws, ss, label_pos_end=True):
    '''
    Performs the sliding window approach on the data and the labels

    return three arrays.
    - data, an array where first dim is the windows
    - labels per window according to end, middle or mode
    - all labels per window

    @param data_x: ids for train
    @param data_y: ids for train
    @param ws: ids for train
    @param ss: ids for train
    @param label_pos_end: ids for train
    '''

    print("Sliding window: Creating windows {} with step {}".format(ws, ss))

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    count_l = 0
    idy = 0
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
                    labels = np.zeros((133 + 1)).astype(int)
                    count_l = np.bincount(sw[:, 1], minlength=len(classes_labels))
                    idy = np.argmax(count_l)
                    attrs = np.sum(sw[:, 2:], axis=0)
                    attrs[attrs > 0] = 1
                    labels[0] = idy
                    labels[1:] = attrs
                    data_y_labels.append(labels)
                data_y_labels = np.asarray(data_y_labels)


            except:
                print("Sliding window: error with the counting {}".format(count_l))
                print("Sliding window: error with the counting {}".format(idy))
                return np.Inf

            # All labels per window
            data_y_all = np.asarray([i[:] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])

    return data_x.astype(np.float32), data_y_labels.astype(np.uint8), data_y_all.astype(np.uint8)


def generate_CSV(csv_dir, type_file, data_dir):
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
            f.append(data_dir + 'seq_{0:07}.pkl'.format(n))

    np.savetxt(csv_dir + type_file, f, delimiter="\n", fmt='%s')

    return f


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
            f.append(data_dir1 + 'seq_{0:07}.pkl'.format(n))

    for dirpath, dirnames, filenames in os.walk(data_dir2):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:07}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')

    return f

def create_dataset():
    '''
    create dataset
    - Segmentation
    - Storing sequences

    @param half: set for creating dataset with half the frequence.
    '''

    subjects = {"S47": "Brownie", "S54": "Brownie", "S13": "Brownie", "S31": "Brownie", "S09": "Brownie",
                "S12": "Sandwich", "S16": "Sandwich", "S25": "Sandwich", "S34": "Sandwich", "S15": "Sandwich",
                "S28": "Eggs", "S08": "Eggs", "S20": "Eggs", "S162": "Eggs", "S50": "Eggs"}

    subjects = {"S47": "Brownie", "S54": "Brownie", "S13": "Brownie", "S31": "Brownie",
                "S12": "Sandwich", "S16": "Sandwich", "S25": "Sandwich", "S34": "Sandwich",
                "S28": "Eggs", "S20": "Eggs", "S162": "Eggs"}

    train_ids = {"S47": "Brownie", "S54": "Brownie", "S13": "Brownie",
                "S12": "Sandwich", "S16": "Sandwich", "S25": "Sandwich",
                "S28": "Eggs", "S20": "Eggs"}
    val_ids = {"S31": "Brownie", "S34": "Sandwich", "S162": "Eggs"}

    test_ids = {"S09": "Brownie", "S15": "Sandwich", "S50": "Eggs"}

    all_data = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    # general_statistics(train_ids)

    #base_directory = '/data2/fmoya/HAR/datasets/mbientlab_50_recordings/'
    base_directory = '/data/fmoya/HAR/datasets/cmu-data/'

    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'

    generate_data(train_ids, sliding_window_length=200, sliding_window_step=12, data_dir=data_dir_train)
    generate_data(val_ids, sliding_window_length=200, sliding_window_step=12, data_dir=data_dir_val)
    generate_data(test_ids, sliding_window_length=200, sliding_window_step=200, data_dir=data_dir_test)
    print("done")

    generate_CSV(base_directory, "train.csv", data_dir_train)
    generate_CSV(base_directory, "val.csv", data_dir_val)
    generate_CSV(base_directory, "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)

    return


if __name__ == '__main__':

    #load_training(data_set = "training")
    #load_training(data_set="test")
    statistics_measurements()
    create_dataset()
    print("Done")