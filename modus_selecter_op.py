'''
Created on May 17, 2019

@author: fmoya
'''

from __future__ import print_function
import logging
import os

from network_user_op import Network_User
import numpy as np

import torch
import time

import xml.etree.ElementTree as ET
from xml.dom import minidom

class Modus_Selecter(object):
    '''
    classdocs
    '''


    def __init__(self, config, exp = None):
        '''
        Constructor
        '''

        logging.info('    Network_selecter: Constructor')
        self.config = config
        logging.info('    Network_selecter: \n{}'.format(config))

        self.exp = exp
        self.network = Network_User(config, self.exp)

        return

    def save(self, acc_test, f1_weighted_test, f1_mean_test, ea_iter=0, type_simple='training', confusion_matrix=0,
             time_iter=0, precisions=0, recalls=0, best_itera=0, acc_attr_test=0, precisions_attr=0, recalls_attr=0 ):
        """
        Save the results of training and testing according to the configuration.
        As training is repeated several times, results are appended, and mean and std of all the repetitions
        are computed.

        @param acc_test: List of accuracies of val or testing
        @param f1_weighted_test: List of F1w of val or testing
        @param f1_mean_test: List of F1m of val or testing
        @param type_simple: Type of experiment
        @param confusion_matrix: Confusion Matrix
        @param time_iter: Time of experiment run
        @param precisions: List of class precisions
        @param recalls: List of class recalls
        @param best_itera: Best evolution iteration
        """
        xml_file_path = self.config['folder_exp'] + self.config['file_suffix']

        xml_root = ET.Element("Experiment_{}".format(self.config["name_counter"]))
        child_network = ET.SubElement(xml_root, "network", dataset=str(self.config['network']))
        child_dataset = ET.SubElement(child_network, "dataset", dataset=str(self.config['dataset']))
        child = ET.SubElement(child_dataset, "usage_modus", usage_modus=str(self.config['usage_modus']))
        #child = ET.SubElement(child_dataset, "dataset_finetuning",
                              #dataset_finetuning=str(self.config['dataset_finetuning']))
        #child = ET.SubElement(child_dataset, "percentages_names",
                              #percentages_names=str(self.config['percentages_names']))
        child = ET.SubElement(child_dataset, "type_simple", type_simple=str(type_simple))
        child = ET.SubElement(child_dataset, "output", output=str(self.config['output']))

        child = ET.SubElement(child_dataset, "lr", lr=str(self.config['lr']))
        child = ET.SubElement(child_dataset, "epochs", epochs=str(self.config['epochs']))
        child = ET.SubElement(child_dataset, "reshape_input", reshape_input=str(self.config["reshape_input"]))
        child = ET.SubElement(child_dataset, "batch_size_train", bsize=str(self.config["batch_size_train"]))

        child = ET.SubElement(child_dataset, "freeze_options", freeze_options=str(self.config['freeze_options']))
        child = ET.SubElement(child_dataset, "time_iter", time_iter=str(time_iter))
        child = ET.SubElement(child_dataset, "best_itera", best_itera=str(best_itera))

        for expi in range(len(acc_test)):
            child = ET.SubElement(child_dataset, "metrics", acc_test=str(acc_test[expi]),
                                  f1_weighted_test=str(f1_weighted_test[expi]),
                                  f1_mean_test=str(f1_mean_test[expi]))
        child = ET.SubElement(child_dataset, "metrics_mean", acc_test_mean=str(np.mean(acc_test)),
                              f1_weighted_test_mean=str(np.mean(f1_weighted_test)),
                              f1_mean_test_mean=str(np.mean(f1_mean_test)))
        child = ET.SubElement(child_dataset, "metrics_std", acc_test_mean=str(np.std(acc_test)),
                              f1_weighted_test_mean=str(np.std(f1_weighted_test)),
                              f1_mean_test_mean=str(np.std(f1_mean_test)))
        child = ET.SubElement(child_dataset, "confusion_matrix_last",
                              confusion_matrix_last=str(confusion_matrix))
        if type_simple == 'training':
            child = ET.SubElement(child_dataset, "precision_mean", precision_mean=str(precisions))
            child = ET.SubElement(child_dataset, "precision_std", precision_std=str(recalls))
            if self.config['output']== 'attribute':
                child = ET.SubElement(child_dataset, "precision_attr_mean", precision_mean=str(precisions_attr))
                child = ET.SubElement(child_dataset, "precision_attr_std", precision_std=str(recalls_attr))
        else:
            if self.config['output']== 'attribute':
                child = ET.SubElement(child_dataset, "acc_attr_mean", precision_mean=str(np.mean(acc_attr_test, axis=0)))
                child = ET.SubElement(child_dataset, "acc_attr_std", precision_std=str(np.std(acc_attr_test, axis=0)))
            child = ET.SubElement(child_dataset, "precision_mean", precision_mean=str(np.mean(precisions, axis=0)))
            child = ET.SubElement(child_dataset, "precision_std", precision_std=str(np.std(precisions, axis=0)))
            child = ET.SubElement(child_dataset, "recall_mean", recall_mean=str(np.mean(recalls, axis=0)))
            child = ET.SubElement(child_dataset, "recall_std", recall_std=str(np.std(recalls, axis=0)))
            if self.config['output']== 'attribute':
                child = ET.SubElement(child_dataset, "precision_attr_mean", precision_mean=str(np.mean(precisions_attr, axis=0)))
                child = ET.SubElement(child_dataset, "precision_attr_std", precision_std=str(np.std(precisions_attr, axis=0)))
                child = ET.SubElement(child_dataset, "recall_attr_mean", recall_mean=str(np.mean(recalls_attr, axis=0)))
                child = ET.SubElement(child_dataset, "recall_attr_std", recall_std=str(np.std(recalls_attr, axis=0)))
            

        xmlstr = minidom.parseString(ET.tostring(xml_root)).toprettyxml(indent="   ")
        with open(xml_file_path, "a") as f:
            f.write(xmlstr)

        print(xmlstr)

        return

    def train(self, itera=1, testing=False):
        """
        Train method. Train network for a certain number of repetitions
        computing the val performance, Testing using test(), saving the performances

        @param itera: training iteration, as training is repeated X number of times
        @param testing: Enabling testing after training
        """

        logging.info('    Network_selecter: Train')

        acc_train_ac = []
        f1_weighted_train_ac = []
        f1_mean_train_ac = []
        precisions_test = []
        recalls_test = []
        if self.config['output']== 'attribute':
            precisions_attr_test = []
            recalls_attr_test = []
        
        
        if testing:
            acc_test_ac = []
            f1_weighted_test_ac = []
            f1_mean_test_ac = []
            if self.config['output']== 'attribute':
                acc_attr_test_ac = []

        #There will be only one iteration
        #As there is not evolution
        for iter_evl in range(itera):
            start_time_train = time.time()
            
            
            # Training the network and obtaining the validation results
            logging.info('    Network_selecter:    Train iter {}'.format(iter_evl))
            results_train, confusion_matrix_train, best_itera, c_pos_val, c_neg_val= self.network.evolution_evaluation(ea_iter=iter_evl)
            #results_train, confusion_matrix_train, best_itera = self.network.evolution_evaluation(ea_iter=iter_evl)
            
            # Appending results for later saving in results file
            acc_train_ac.append(results_train['acc'])
            f1_weighted_train_ac.append(results_train['f1_weighted'])
            f1_mean_train_ac.append(results_train['f1_mean'])

            time_train = time.time() - start_time_train

            
            if self.config['output']== 'softmax':
                logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                         'f1_weighted {}, f1_mean {}'.format(time_train, results_train['acc'],
                                                             results_train['f1_weighted'], results_train['f1_mean']))
            elif self.config['output']== 'attribute':
                logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                         'f1_weighted {}, f1_mean {}, acc_attr {}'.format(time_train, results_train['acc'],
                                                             results_train['f1_weighted'], results_train['f1_mean'], results_train['acc_attrs'] ))
            
            
            self.exp.log_scalar("accuracy_train_mo_{}".format(iter_evl),results_train['acc'])
            self.exp.log_scalar("f1_w_train_mo_{}".format(iter_evl),results_train['f1_weighted'])
            self.exp.log_scalar("f1_m_train_mo_{}".format(iter_evl), results_train['f1_mean'])
            self.exp.log_scalar("best_iter_{}".format(iter_evl), best_itera)
            if self.config['output']== 'attribute':
                p=results_train['acc_attrs']
                for i in range(0,p.shape[0]):
                    self.exp.log_scalar("acc_attr_{}_train_mo_{}".format(i, iter_evl),p[i])
            
            if c_pos_val[0] == 0:
                self.exp.log_scalar("none_p_v_f{}".format(iter_evl), c_pos_val[0])
            else:
                self.exp.log_scalar("none_p_v_f{}".format(iter_evl), c_pos_val[0]/(c_pos_val[0]+c_neg_val[0]))
            if c_pos_val[1] == 0:
                self.exp.log_scalar("lying_p_v_f{}".format(iter_evl), c_pos_val[1])
            else:
                self.exp.log_scalar("lying_p_v_f{}".format(iter_evl), c_pos_val[1]/(c_pos_val[1]+c_neg_val[1]))
            if c_pos_val[2] == 0:
                self.exp.log_scalar("sitting_p_v_f{}".format(iter_evl), c_pos_val[2])
            else:
                self.exp.log_scalar("sitting_p_v_f{}".format(iter_evl), c_pos_val[2]/(c_pos_val[2]+c_neg_val[2]))
            if c_pos_val[3] == 0:
                self.exp.log_scalar("standing_p_v_f{}".format(iter_evl), c_pos_val[3])
            else:
                self.exp.log_scalar("standing_p_v_f{}".format(iter_evl), c_pos_val[3]/(c_pos_val[3]+c_neg_val[3]))
            if c_pos_val[4] == 0:
                self.exp.log_scalar("walking_p_v_f{}".format(iter_evl), c_pos_val[4])
            else:
                self.exp.log_scalar("walking_p_v_f{}".format(iter_evl), c_pos_val[4]/(c_pos_val[4]+c_neg_val[4]))
            if c_pos_val[5] == 0:
                self.exp.log_scalar("running_p_v_f{}".format(iter_evl), c_pos_val[5])
            else:
                self.exp.log_scalar("running_p_v_f{}".format(iter_evl), c_pos_val[5]/(c_pos_val[5]+c_neg_val[5]))
            if c_pos_val[6] == 0:
                self.exp.log_scalar("cycling_p_v_f{}".format(iter_evl), c_pos_val[6])
            else:
                self.exp.log_scalar("cycling_p_v_f{}".format(iter_evl), c_pos_val[6]/(c_pos_val[6]+c_neg_val[6]))
            if c_pos_val[7] == 0:
                self.exp.log_scalar("nordicwalk_p_v_f{}".format(iter_evl), c_pos_val[7])
            else:
                self.exp.log_scalar("nordicwalk_p_v_f{}".format(iter_evl), c_pos_val[7]/(c_pos_val[7]+c_neg_val[7]))
            if c_pos_val[8] == 0:
                self.exp.log_scalar("ascending_p_v_f{}".format(iter_evl), c_pos_val[8])
            else:
                self.exp.log_scalar("ascending_p_v_f{}".format(iter_evl), c_pos_val[8]/(c_pos_val[8]+c_neg_val[8]))
            if c_pos_val[9] == 0:
                self.exp.log_scalar("descending_p_v_f{}".format(iter_evl), c_pos_val[9])
            else:
                self.exp.log_scalar("descending_p_v_f{}".format(iter_evl), c_pos_val[9]/(c_pos_val[9]+c_neg_val[9]))
            if c_pos_val[10] == 0:
                self.exp.log_scalar("vaccum_p_v_f{}".format(iter_evl), c_pos_val[10])
            else:
                self.exp.log_scalar("vaccum_p_v_f{}".format(iter_evl), c_pos_val[10]/(c_pos_val[10]+c_neg_val[10]))
            if c_pos_val[11] == 0:
                self.exp.log_scalar("ropejumping_p_v_f{}".format(iter_evl), c_pos_val[11])
            else:
                self.exp.log_scalar("ropejumping_p_v_f{}".format(iter_evl), c_pos_val[11]/(c_pos_val[11]+c_neg_val[11]))
                    
            if c_neg_val[0] == 0:
                self.exp.log_scalar("none_n_v_f{}".format(iter_evl), c_neg_val[0])
            else:
                self.exp.log_scalar("none_n_v_f{}".format(iter_evl), c_neg_val[0]/(c_pos_val[0]+c_neg_val[0]))
            if c_neg_val[1] == 0:
                self.exp.log_scalar("lying_n_v_f{}".format(iter_evl), c_neg_val[1])
            else:
                self.exp.log_scalar("lying_n_v_f{}".format(iter_evl), c_neg_val[1]/(c_pos_val[1]+c_neg_val[1]))
            if c_neg_val[2] == 0:
                self.exp.log_scalar("sitting_n_v_f{}".format(iter_evl), c_neg_val[2])
            else:
                self.exp.log_scalar("sitting_n_v_f{}".format(iter_evl), c_neg_val[2]/(c_pos_val[2]+c_neg_val[2]))
            if c_neg_val[3] == 0:
                self.exp.log_scalar("standing_n_v_f{}".format(iter_evl), c_neg_val[3])
            else:
                self.exp.log_scalar("standing_n_v_f{}".format(iter_evl), c_neg_val[3]/(c_pos_val[3]+c_neg_val[3]))
            if c_neg_val[4] == 0:
                self.exp.log_scalar("walking_n_v_f{}".format(iter_evl), c_neg_val[4])
            else:
                self.exp.log_scalar("walking_n_v_f{}".format(iter_evl), c_neg_val[4]/(c_pos_val[4]+c_neg_val[4]))
            if c_neg_val[5] == 0:
                self.exp.log_scalar("running_n_v_f{}".format(iter_evl), c_neg_val[5])
            else:
                self.exp.log_scalar("running_n_v_f{}".format(iter_evl), c_neg_val[5]/(c_pos_val[5]+c_neg_val[5]))
            if c_neg_val[6] == 0:
                self.exp.log_scalar("cycling_n_v_f{}".format(iter_evl), c_neg_val[6])
            else:
                self.exp.log_scalar("cycling_n_v_f{}".format(iter_evl), c_neg_val[6]/(c_pos_val[6]+c_neg_val[6]))
            if c_neg_val[7] == 0:
                self.exp.log_scalar("nordicwalk_n_v_f{}".format(iter_evl), c_neg_val[7])
            else:
                self.exp.log_scalar("nordicwalk_n_v_f{}".format(iter_evl), c_neg_val[7]/(c_pos_val[7]+c_neg_val[7]))
            if c_neg_val[8] == 0:
                self.exp.log_scalar("ascending_n_v_f{}".format(iter_evl), c_neg_val[8])
            else:
                self.exp.log_scalar("ascending_n_v_f{}".format(iter_evl), c_neg_val[8]/(c_pos_val[8]+c_neg_val[8]))
            if c_neg_val[9] == 0:
                self.exp.log_scalar("descending_n_v_f{}".format(iter_evl), c_neg_val[9])
            else:
                self.exp.log_scalar("descending_n_v_f{}".format(iter_evl), c_neg_val[9]/(c_pos_val[9]+c_neg_val[9]))
            if c_neg_val[10] == 0:
                self.exp.log_scalar("vaccum_n_v_f{}".format(iter_evl), c_neg_val[10])
            else:
                self.exp.log_scalar("vaccum_n_v_f{}".format(iter_evl), c_neg_val[10]/(c_pos_val[10]+c_neg_val[10]))
            if c_neg_val[11] == 0:
                self.exp.log_scalar("ropejumping_n_v_f{}".format(iter_evl), c_neg_val[11])
            else:
                self.exp.log_scalar("ropejumping_n_v_f{}".format(iter_evl), c_neg_val[11]/(c_pos_val[11]+c_neg_val[11]))
            
            '''
            if self.config['dataset']=='locomotion':
                if c_pos_val[0] == 0:
                    self.exp.log_scalar("stand_p_v_f{}".format(iter_evl), c_pos_val[0])
                else:
                    self.exp.log_scalar("stand_p_v_f{}".format(iter_evl), c_pos_val[0]/(c_pos_val[0]+c_neg_val[0]))
                if c_pos_val[1] == 0:
                       self.exp.log_scalar("walk_p_v_f{}".format(iter_evl), c_pos_val[1])
                else:
                    self.exp.log_scalar("walk_p_v_f{}".format(iter_evl), c_pos_val[1]/(c_pos_val[1]+c_neg_val[1]))
                if c_pos_val[2] == 0:
                    self.exp.log_scalar("sit_p_v_f{}".format(iter_evl), c_pos_val[2])
                else:
                    self.exp.log_scalar("sit_p_v_f{}".format(iter_evl), c_pos_val[2]/(c_pos_val[2]+c_neg_val[2]))
                if c_pos_val[3] == 0:
                    self.exp.log_scalar("lie_p_v_f{}".format(iter_evl), c_pos_val[3])
                else:
                    self.exp.log_scalar("lie_p_v_f{}".format(iter_evl), c_pos_val[3]/(c_pos_val[3]+c_neg_val[3]))
                if c_pos_val[4] == 0:
                    self.exp.log_scalar("none_p_v_f{}".format(iter_evl), c_pos_val[4])
                else:
                    self.exp.log_scalar("none_p_v_f{}".format(iter_evl), c_pos_val[4]/(c_pos_val[4]+c_neg_val[4]))
            
                if c_neg_val[0] == 0:
                    self.exp.log_scalar("stand_n_v_f{}".format(iter_evl), c_neg_val[0])
                else:
                    self.exp.log_scalar("stand_n_v_f{}".format(iter_evl), c_neg_val[0]/(c_pos_val[0]+c_neg_val[0]))
                if c_neg_val[1] == 0:
                    self.exp.log_scalar("walk_n_v_f{}".format(iter_evl), c_neg_val[1])
                else:
                    self.exp.log_scalar("walk_n_v_f{}".format(iter_evl), c_neg_val[1]/(c_pos_val[1]+c_neg_val[1]))
                if c_neg_val[2] == 0:
                    self.exp.log_scalar("sit_n_v_f{}".format(iter_evl), c_neg_val[2])
                else:
                    self.exp.log_scalar("sit_n_v_f{}".format(iter_evl), c_neg_val[2]/(c_pos_val[2]+c_neg_val[2]))
                if c_neg_val[3] == 0:
                    self.exp.log_scalar("lie_n_v_f{}".format(iter_evl), c_neg_val[3])
                else:
                    self.exp.log_scalar("lie_n_v_f{}".format(iter_evl), c_neg_val[3]/(c_pos_val[3]+c_neg_val[3]))
                if c_neg_val[4] == 0:
                    self.exp.log_scalar("none_n_v_f{}".format(iter_evl), c_neg_val[4])
                else:
                    self.exp.log_scalar("none_n_v_f{}".format(iter_evl), c_neg_val[4]/(c_pos_val[4]+c_neg_val[4]))
              
            elif self.config['dataset']=='gesture':
                if c_pos_val[0] == 0:
                    self.exp.log_scalar("opendoor1_p_v_f{}".format(iter_evl), c_pos_val[0])
                else:
                    self.exp.log_scalar("opendoor1_p_v_f{}".format(iter_evl), c_pos_val[0]/(c_pos_val[0]+c_neg_val[0]))
                if c_pos_val[1] == 0:
                    self.exp.log_scalar("opendoor2_p_v_f{}".format(iter_evl), c_pos_val[1])
                else:
                    self.exp.log_scalar("opendoor2_p_v_f{}".format(iter_evl), c_pos_val[1]/(c_pos_val[1]+c_neg_val[1]))
                if c_pos_val[2] == 0:
                    self.exp.log_scalar("closedoor1_p_v_f{}".format(iter_evl), c_pos_val[2])
                else:
                    self.exp.log_scalar("closedoor1_p_v_f{}".format(iter_evl), c_pos_val[2]/(c_pos_val[2]+c_neg_val[2]))
                if c_pos_val[3] == 0:
                    self.exp.log_scalar("closedoor2_p_v_f{}".format(iter_evl), c_pos_val[3])
                else:
                    self.exp.log_scalar("closedoor2_p_v_f{}".format(iter_evl), c_pos_val[3]/(c_pos_val[3]+c_neg_val[3]))
                if c_pos_val[4] == 0:
                    self.exp.log_scalar("openfridge_p_v_f{}".format(iter_evl), c_pos_val[4])
                else:
                    self.exp.log_scalar("openfridge_p_v_f{}".format(iter_evl), c_pos_val[4]/(c_pos_val[4]+c_neg_val[4]))
                if c_pos_val[5] == 0:
                    self.exp.log_scalar("closefridge_p_v_f{}".format(iter_evl), c_pos_val[5])
                else:
                    self.exp.log_scalar("closefridge_p_v_f{}".format(iter_evl), c_pos_val[5]/(c_pos_val[5]+c_neg_val[5]))
                if c_pos_val[6] == 0:
                    self.exp.log_scalar("openDW_p_v_f{}".format(iter_evl), c_pos_val[6])
                else:
                    self.exp.log_scalar("openDW_p_v_f{}".format(iter_evl), c_pos_val[6]/(c_pos_val[6]+c_neg_val[6]))
                if c_pos_val[7] == 0:
                    self.exp.log_scalar("closeDW_p_v_f{}".format(iter_evl), c_pos_val[7])
                else:
                    self.exp.log_scalar("closeDW_p_v_f{}".format(iter_evl), c_pos_val[7]/(c_pos_val[7]+c_neg_val[7]))
                if c_pos_val[8] == 0:
                    self.exp.log_scalar("opendrawer1_p_v_f{}".format(iter_evl), c_pos_val[8])
                else:
                    self.exp.log_scalar("opendrawer1_p_v_f{}".format(iter_evl), c_pos_val[8]/(c_pos_val[8]+c_neg_val[8]))
                if c_pos_val[9] == 0:
                    self.exp.log_scalar("closedrawer1_p_v_f{}".format(iter_evl), c_pos_val[9])
                else:
                    self.exp.log_scalar("closedrawer1_p_v_f{}".format(iter_evl), c_pos_val[9]/(c_pos_val[9]+c_neg_val[9]))
                if c_pos_val[10] == 0:
                    self.exp.log_scalar("opendrawer2_p_v_f{}".format(iter_evl), c_pos_val[10])
                else:
                    self.exp.log_scalar("opendrawer2_p_v_f{}".format(iter_evl), c_pos_val[10]/(c_pos_val[10]+c_neg_val[10]))
                if c_pos_val[11] == 0:
                    self.exp.log_scalar("closedrawer2_p_v_f{}".format(iter_evl), c_pos_val[11])
                else:
                    self.exp.log_scalar("closedrawer2_p_v_f{}".format(iter_evl), c_pos_val[11]/(c_pos_val[11]+c_neg_val[11]))
                if c_pos_val[12] == 0:
                    self.exp.log_scalar("opendrawer3_p_v_f{}".format(iter_evl), c_pos_val[12])
                else:
                    self.exp.log_scalar("opendrawer3_p_v_f{}".format(iter_evl), c_pos_val[12]/(c_pos_val[12]+c_neg_val[12]))
                if c_pos_val[13] == 0:
                    self.exp.log_scalar("closedrawer3_p_v_f{}".format(iter_evl), c_pos_val[13])
                else:
                    self.exp.log_scalar("closedrawer3_p_v_f{}".format(iter_evl), c_pos_val[13]/(c_pos_val[13]+c_neg_val[13]))
                if c_pos_val[14] == 0:
                    self.exp.log_scalar("cleantable_p_v_f{}".format(iter_evl), c_pos_val[14])
                else:
                    self.exp.log_scalar("cleantable_p_v_f{}".format(iter_evl), c_pos_val[14]/(c_pos_val[14]+c_neg_val[14]))
                if c_pos_val[15] == 0:
                    self.exp.log_scalar("drinkcup_p_v_f{}".format(iter_evl), c_pos_val[15])
                else:
                    self.exp.log_scalar("drinkcup_p_v_f{}".format(iter_evl), c_pos_val[15]/(c_pos_val[15]+c_neg_val[15]))
                if c_pos_val[16] == 0:
                    self.exp.log_scalar("toggle_p_v_f{}".format(iter_evl), c_pos_val[16])
                else:
                    self.exp.log_scalar("toggle_p_v_f{}".format(iter_evl), c_pos_val[16]/(c_pos_val[16]+c_neg_val[16]))
                if c_pos_val[17] == 0:
                    self.exp.log_scalar("none_p_v_f{}".format(iter_evl), c_pos_val[17])
                else:
                    self.exp.log_scalar("none_p_v_f{}".format(iter_evl), c_pos_val[17]/(c_pos_val[17]+c_neg_val[17]))
                    
                if c_neg_val[0] == 0:
                    self.exp.log_scalar("opendoor1_n_v_f{}".format(iter_evl), c_neg_val[0])
                else:
                    self.exp.log_scalar("opendoor1_n_v_f{}".format(iter_evl), c_neg_val[0]/(c_pos_val[0]+c_neg_val[0]))
                if c_neg_val[1] == 0:
                    self.exp.log_scalar("opendoor2_n_v_f{}".format(iter_evl), c_neg_val[1])
                else:
                    self.exp.log_scalar("opendoor2_n_v_f{}".format(iter_evl), c_neg_val[1]/(c_pos_val[1]+c_neg_val[1]))
                if c_neg_val[2] == 0:
                    self.exp.log_scalar("closedoor1_n_v_f{}".format(iter_evl), c_neg_val[2])
                else:
                    self.exp.log_scalar("closedoor1_n_v_f{}".format(iter_evl), c_neg_val[2]/(c_pos_val[2]+c_neg_val[2]))
                if c_neg_val[3] == 0:
                    self.exp.log_scalar("closedoor2_n_v_f{}".format(iter_evl), c_neg_val[3])
                else:
                    self.exp.log_scalar("closedoor2_n_v_f{}".format(iter_evl), c_neg_val[3]/(c_pos_val[3]+c_neg_val[3]))
                if c_neg_val[4] == 0:
                    self.exp.log_scalar("openfridge_n_v_f{}".format(iter_evl), c_neg_val[4])
                else:
                    self.exp.log_scalar("openfridge_n_v_f{}".format(iter_evl), c_neg_val[4]/(c_pos_val[4]+c_neg_val[4]))
                if c_neg_val[5] == 0:
                    self.exp.log_scalar("closefridge_n_v_f{}".format(iter_evl), c_neg_val[5])
                else:
                    self.exp.log_scalar("closefridge_n_v_f{}".format(iter_evl), c_neg_val[5]/(c_pos_val[5]+c_neg_val[5]))
                if c_neg_val[6] == 0:
                    self.exp.log_scalar("openDW_n_v_f{}".format(iter_evl), c_neg_val[6])
                else:
                    self.exp.log_scalar("openDW_n_v_f{}".format(iter_evl), c_neg_val[6]/(c_pos_val[6]+c_neg_val[6]))
                if c_neg_val[7] == 0:
                    self.exp.log_scalar("closeDW_n_v_f{}".format(iter_evl), c_neg_val[7])
                else:
                    self.exp.log_scalar("closeDW_n_v_f{}".format(iter_evl), c_neg_val[7]/(c_pos_val[7]+c_neg_val[7]))
                if c_neg_val[8] == 0:
                    self.exp.log_scalar("opendrawer1_n_v_f{}".format(iter_evl), c_neg_val[8])
                else:
                    self.exp.log_scalar("opendrawer1_n_v_f{}".format(iter_evl), c_neg_val[8]/(c_pos_val[8]+c_neg_val[8]))
                if c_neg_val[9] == 0:
                    self.exp.log_scalar("closedrawer1_n_v_f{}".format(iter_evl), c_neg_val[9])
                else:
                    self.exp.log_scalar("closedrawer1_n_v_f{}".format(iter_evl), c_neg_val[9]/(c_pos_val[9]+c_neg_val[9]))
                if c_neg_val[10] == 0:
                    self.exp.log_scalar("opendrawer2_n_v_f{}".format(iter_evl), c_neg_val[10])
                else:
                    self.exp.log_scalar("opendrawer2_n_v_f{}".format(iter_evl), c_neg_val[10]/(c_pos_val[10]+c_neg_val[10]))
                if c_neg_val[11] == 0:
                    self.exp.log_scalar("closedrawer2_n_v_f{}".format(iter_evl), c_neg_val[11])
                else:
                    self.exp.log_scalar("closedrawer2_n_v_f{}".format(iter_evl), c_neg_val[11]/(c_pos_val[11]+c_neg_val[11]))
                if c_neg_val[12] == 0:
                    self.exp.log_scalar("opendrawer3_n_v_f{}".format(iter_evl), c_neg_val[12])
                else:
                    self.exp.log_scalar("opendrawer3_n_v_f{}".format(iter_evl), c_neg_val[12]/(c_pos_val[12]+c_neg_val[12]))
                if c_neg_val[13] == 0:
                    self.exp.log_scalar("closedrawer3_n_v_f{}".format(iter_evl), c_neg_val[13])
                else:
                    self.exp.log_scalar("closedrawer3_n_v_f{}".format(iter_evl), c_neg_val[13]/(c_pos_val[13]+c_neg_val[13]))
                if c_neg_val[14] == 0:
                    self.exp.log_scalar("cleantable_n_v_f{}".format(iter_evl), c_neg_val[14])
                else:
                    self.exp.log_scalar("cleantable_n_v_f{}".format(iter_evl), c_neg_val[14]/(c_pos_val[14]+c_neg_val[14]))
                if c_neg_val[15] == 0:
                    self.exp.log_scalar("drinkcup_n_v_f{}".format(iter_evl), c_neg_val[15])
                else:
                    self.exp.log_scalar("drinkcup_n_v_f{}".format(iter_evl), c_neg_val[15]/(c_pos_val[15]+c_neg_val[15]))
                if c_neg_val[16] == 0:
                    self.exp.log_scalar("toggle_n_v_f{}".format(iter_evl), c_neg_val[16])
                else:
                    self.exp.log_scalar("toggle_n_v_f{}".format(iter_evl), c_neg_val[16]/(c_pos_val[16]+c_neg_val[16]))
                if c_neg_val[17] == 0:
                    self.exp.log_scalar("none_n_v_f{}".format(iter_evl), c_neg_val[17])
                else:
                    self.exp.log_scalar("none_n_v_f{}".format(iter_evl), c_neg_val[17]/(c_pos_val[17]+c_neg_val[17]))
                
                '''
                           
            # Saving the results
            self.save(acc_train_ac, f1_weighted_train_ac, f1_mean_train_ac, time_iter=time_train, precisions=results_train['precision'], 
                      recalls=results_train['recall'], best_itera=best_itera)
            
            # Testing the network
            if testing:
                start_time_test = time.time()
                results_test, confusion_matrix_test, count_pos_test, count_neg_test = self.test(testing=True)
                #results_test, confusion_matrix_test= self.test(testing=True)
                acc_test_ac.append(results_test['acc'])
                f1_weighted_test_ac.append(results_test['f1_weighted'])
                f1_mean_test_ac.append(results_test['f1_mean'])
                precisions_test.append(results_test['precision'].numpy())
                recalls_test.append(results_test['recall'].numpy())
                if self.config['output']== 'attribute':
                    acc_attr_test_ac.append(results_test['acc_attrs'])
                    precisions_attr_test.append(results_test['precision_attr'].numpy())
                    recalls_attr_test.append(results_test['recall_attr'].numpy())
                
                time_test = time.time() - start_time_test

        if testing:
            if self.config['output']== 'softmax':
                self.save(acc_test_ac, f1_weighted_test_ac, f1_mean_test_ac, type_simple='testing',
                      confusion_matrix=confusion_matrix_test, time_iter=time_test, precisions=np.array(precisions_test),
                      recalls=np.array(recalls_test), best_itera=0)
            elif self.config['output']== 'attribute':
                self.save(acc_test_ac, f1_weighted_test_ac, f1_mean_test_ac, type_simple='testing',
                      confusion_matrix=confusion_matrix_test, time_iter=time_test, precisions=np.array(precisions_test),
                      recalls=np.array(recalls_test), best_itera=0, acc_attr_test=acc_attr_test_ac, precisions_attr=np.array(precisions_attr_test), 
                      recalls_attr=np.array(recalls_attr_test))
           
            self.exp.log_scalar("accuracy_test_mo_{}".format(iter_evl),results_test['acc'])
            self.exp.log_scalar("f1_w_test_mo_{}".format(iter_evl),results_test['f1_weighted'])
            self.exp.log_scalar("f1_m_test_mo_{}".format(iter_evl),results_test['f1_mean'])
            
            if self.config['output']== 'attribute':
                p=results_test['acc_attrs']
                for i in range(0,p.shape[0]):
                    self.exp.log_scalar("acc_attr_{}_test_mo_{}".format(i, iter_evl),p[i])
                  
            if count_pos_test[0] == 0:
                self.exp.log_scalar("none_p_test{}".format(iter_evl), count_pos_test[0])
            else:
                self.exp.log_scalar("none_p_test{}".format(iter_evl), count_pos_test[0]/(count_pos_test[0]+count_neg_test[0]))
            if count_pos_test[1] == 0:
                self.exp.log_scalar("lying_p_test{}".format(iter_evl), count_pos_test[1])
            else:
                self.exp.log_scalar("lying_p_test{}".format(iter_evl), count_pos_test[1]/(count_pos_test[1]+count_neg_test[1]))
            if count_pos_test[2] == 0:
                self.exp.log_scalar("sitting_p_test{}".format(iter_evl), count_pos_test[2])
            else:
                self.exp.log_scalar("sitting_p_test{}".format(iter_evl), count_pos_test[2]/(count_pos_test[2]+count_neg_test[2]))
            if count_pos_test[3] == 0:
                self.exp.log_scalar("standing_p_test{}".format(iter_evl), count_pos_test[3])
            else:
                self.exp.log_scalar("standing_p_test{}".format(iter_evl), count_pos_test[3]/(count_pos_test[3]+count_neg_test[3]))
            if count_pos_test[4] == 0:
                self.exp.log_scalar("walking_p_test{}".format(iter_evl), count_pos_test[4])
            else:
                self.exp.log_scalar("walking_p_test{}".format(iter_evl), count_pos_test[4]/(count_pos_test[4]+count_neg_test[4]))
            if count_pos_test[5] == 0:
                self.exp.log_scalar("running_p_test{}".format(iter_evl), count_pos_test[5])
            else:
                self.exp.log_scalar("running_p_test{}".format(iter_evl), count_pos_test[5]/(count_pos_test[5]+count_neg_test[5]))
            if count_pos_test[6] == 0:
                self.exp.log_scalar("cycling_p_test{}".format(iter_evl), count_pos_test[6])
            else:
                self.exp.log_scalar("cycling_p_test{}".format(iter_evl), count_pos_test[6]/(count_pos_test[6]+count_neg_test[6]))
            if count_pos_test[7] == 0:
                self.exp.log_scalar("nordicwalk_p_test{}".format(iter_evl), count_pos_test[7])
            else:
                self.exp.log_scalar("nordicwalk_p_test{}".format(iter_evl), count_pos_test[7]/(count_pos_test[7]+count_neg_test[7]))
            if count_pos_test[8] == 0:
                self.exp.log_scalar("ascending_p_test{}".format(iter_evl), count_pos_test[8])
            else:
                self.exp.log_scalar("ascending_p_test{}".format(iter_evl), count_pos_test[8]/(count_pos_test[8]+count_neg_test[8]))
            if count_pos_test[9] == 0:
                self.exp.log_scalar("descending_p_test{}".format(iter_evl), count_pos_test[9])
            else:
                self.exp.log_scalar("descending_p_test{}".format(iter_evl), count_pos_test[9]/(count_pos_test[9]+count_neg_test[9]))
            if count_pos_test[10] == 0:
                self.exp.log_scalar("vaccum_p_test{}".format(iter_evl), count_pos_test[10])
            else:
                self.exp.log_scalar("vaccum_p_test{}".format(iter_evl), count_pos_test[10]/(count_pos_test[10]+count_neg_test[10]))
            if count_pos_test[11] == 0:
                self.exp.log_scalar("ropejumping_p_test{}".format(iter_evl), count_pos_test[11])
            else:
                self.exp.log_scalar("ropejumping_p_test{}".format(iter_evl), count_pos_test[11]/(count_pos_test[11]+count_neg_test[11]))
            
            if count_neg_test[0] == 0:
                self.exp.log_scalar("none_n_test{}".format(iter_evl), count_neg_test[0])
            else:
                self.exp.log_scalar("none_n_test{}".format(iter_evl), count_neg_test[0]/(count_pos_test[0]+count_neg_test[0]))
            if count_neg_test[1] == 0:
                self.exp.log_scalar("lying_n_test{}".format(iter_evl), count_neg_test[1])
            else:
                self.exp.log_scalar("lying_n_test{}".format(iter_evl), count_neg_test[1]/(count_pos_test[1]+count_neg_test[1]))
            if count_neg_test[2] == 0:
                self.exp.log_scalar("sitting_n_test{}".format(iter_evl), count_neg_test[2])
            else:
                self.exp.log_scalar("sitting_n_test{}".format(iter_evl), count_neg_test[2]/(count_pos_test[2]+count_neg_test[2]))
            if count_neg_test[3] == 0:
                self.exp.log_scalar("standing_n_test{}".format(iter_evl), count_neg_test[3])
            else:
                self.exp.log_scalar("standing_n_test{}".format(iter_evl), count_neg_test[3]/(count_pos_test[3]+count_neg_test[3]))
            if count_neg_test[4] == 0:
                self.exp.log_scalar("walking_n_test{}".format(iter_evl), count_neg_test[4])
            else:
                self.exp.log_scalar("walking_n_test{}".format(iter_evl), count_neg_test[4]/(count_pos_test[4]+count_neg_test[4]))
            if count_neg_test[5] == 0:
                self.exp.log_scalar("running_n_test{}".format(iter_evl), count_neg_test[5])
            else:
                self.exp.log_scalar("running_n_test{}".format(iter_evl), count_neg_test[5]/(count_pos_test[5]+count_neg_test[5]))
            if count_neg_test[6] == 0:
                self.exp.log_scalar("cycling_n_test{}".format(iter_evl), count_neg_test[6])
            else:
                self.exp.log_scalar("cycling_n_test{}".format(iter_evl), count_neg_test[6]/(count_pos_test[6]+count_neg_test[6]))
            if count_neg_test[7] == 0:
                self.exp.log_scalar("nordicwalk_n_test{}".format(iter_evl), count_neg_test[7])
            else:
                self.exp.log_scalar("nordicwalk_n_test{}".format(iter_evl), count_neg_test[7]/(count_pos_test[7]+count_neg_test[7]))
            if count_neg_test[8] == 0:
                self.exp.log_scalar("ascending_n_test{}".format(iter_evl), count_neg_test[8])
            else:
                self.exp.log_scalar("ascending_n_test{}".format(iter_evl), count_neg_test[8]/(count_pos_test[8]+count_neg_test[8]))
            if count_neg_test[9] == 0:
                self.exp.log_scalar("descending_n_test{}".format(iter_evl), count_neg_test[9])
            else:
                self.exp.log_scalar("descending_n_test{}".format(iter_evl), count_neg_test[9]/(count_pos_test[9]+count_neg_test[9]))
            if count_neg_test[10] == 0:
                self.exp.log_scalar("vaccum_n_test{}".format(iter_evl), count_neg_test[10])
            else:
                self.exp.log_scalar("vaccum_n_test{}".format(iter_evl), count_neg_test[10]/(count_pos_test[10]+count_neg_test[10]))
            if count_neg_test[11] == 0:
                self.exp.log_scalar("ropejumping_n_test{}".format(iter_evl), count_neg_test[11])
            else:
                self.exp.log_scalar("ropejumping_n_test{}".format(iter_evl), count_neg_test[11]/(count_pos_test[11]+count_neg_test[11]))
            
            '''
            if self.config['dataset']=='locomotion':
                if count_pos_test[0] == 0:
                    self.exp.log_scalar("stand_p_test{}".format(iter_evl), count_pos_test[0])
                else:
                    self.exp.log_scalar("stand_p_test{}".format(iter_evl), count_pos_test[0]/(count_pos_test[0]+count_neg_test[0]))
                if count_pos_test[1] == 0:
                       self.exp.log_scalar("walk_p_test{}".format(iter_evl), count_pos_test[1])
                else:
                    self.exp.log_scalar("walk_p_test{}".format(iter_evl), count_pos_test[1]/(count_pos_test[1]+count_neg_test[1]))
                if count_pos_test[2] == 0:
                    self.exp.log_scalar("sit_p_test{}".format(iter_evl), count_pos_test[2])
                else:
                    self.exp.log_scalar("sit_p_test{}".format(iter_evl), count_pos_test[2]/(count_pos_test[2]+count_neg_test[2]))
                if count_pos_test[3] == 0:
                    self.exp.log_scalar("lie_p_test{}".format(iter_evl), count_pos_test[3])
                else:
                    self.exp.log_scalar("lie_p_test{}".format(iter_evl), count_pos_test[3]/(count_pos_test[3]+count_neg_test[3]))
                if count_pos_test[4] == 0:
                    self.exp.log_scalar("none_p_test{}".format(iter_evl), count_pos_test[4])
                else:
                    self.exp.log_scalar("none_p_test{}".format(iter_evl), count_pos_test[4]/(count_pos_test[4]+count_neg_test[4]))
            
                if count_neg_test[0] == 0:
                    self.exp.log_scalar("stand_n_test{}".format(iter_evl), count_neg_test[0])
                else:
                    self.exp.log_scalar("stand_n_test{}".format(iter_evl), count_neg_test[0]/(count_pos_test[0]+count_neg_test[0]))
                if count_neg_test[1] == 0:
                    self.exp.log_scalar("walk_n_test{}".format(iter_evl), count_neg_test[1])
                else:
                    self.exp.log_scalar("walk_n_test{}".format(iter_evl), count_neg_test[1]/(count_pos_test[1]+count_neg_test[1]))
                if count_neg_test[2] == 0:
                    self.exp.log_scalar("sit_n_test{}".format(iter_evl), count_neg_test[2])
                else:
                    self.exp.log_scalar("sit_n_test{}".format(iter_evl), count_neg_test[2]/(count_pos_test[2]+count_neg_test[2]))
                if count_neg_test[3] == 0:
                    self.exp.log_scalar("lie_n_test{}".format(iter_evl), count_neg_test[3])
                else:
                    self.exp.log_scalar("lie_n_test{}".format(iter_evl), count_neg_test[3]/(count_pos_test[3]+count_neg_test[3]))
                if count_neg_test[4] == 0:
                    self.exp.log_scalar("none_n_test{}".format(iter_evl), count_neg_test[4])
                else:
                    self.exp.log_scalar("none_n_test{}".format(iter_evl), count_neg_test[4]/(count_pos_test[4]+count_neg_test[4]))
              
            elif self.config['dataset']=='gesture':
                if count_pos_test[0] == 0:
                    self.exp.log_scalar("opendoor1_p_test{}".format(iter_evl), count_pos_test[0])
                else:
                    self.exp.log_scalar("opendoor1_p_test{}".format(iter_evl), count_pos_test[0]/(count_pos_test[0]+count_neg_test[0]))
                if count_pos_test[1] == 0:
                    self.exp.log_scalar("opendoor2_p_test{}".format(iter_evl), count_pos_test[1])
                else:
                    self.exp.log_scalar("opendoor2_p_test{}".format(iter_evl), count_pos_test[1]/(count_pos_test[1]+count_neg_test[1]))
                if count_pos_test[2] == 0:
                    self.exp.log_scalar("closedoor1_p_test{}".format(iter_evl), count_pos_test[2])
                else:
                    self.exp.log_scalar("closedoor1_p_test{}".format(iter_evl), count_pos_test[2]/(count_pos_test[2]+count_neg_test[2]))
                if count_pos_test[3] == 0:
                    self.exp.log_scalar("closedoor2_p_test{}".format(iter_evl), count_pos_test[3])
                else:
                    self.exp.log_scalar("closedoor2_p_test{}".format(iter_evl), count_pos_test[3]/(count_pos_test[3]+count_neg_test[3]))
                if count_pos_test[4] == 0:
                    self.exp.log_scalar("openfridge_p_test{}".format(iter_evl), count_pos_test[4])
                else:
                    self.exp.log_scalar("openfridge_p_test{}".format(iter_evl), count_pos_test[4]/(count_pos_test[4]+count_neg_test[4]))
                if count_pos_test[5] == 0:
                    self.exp.log_scalar("closefridge_p_test{}".format(iter_evl), count_pos_test[5])
                else:
                    self.exp.log_scalar("closefridge_p_test{}".format(iter_evl), count_pos_test[5]/(count_pos_test[5]+count_neg_test[5]))
                if count_pos_test[6] == 0:
                    self.exp.log_scalar("openDW_p_test{}".format(iter_evl), count_pos_test[6])
                else:
                    self.exp.log_scalar("openDW_p_test{}".format(iter_evl), count_pos_test[6]/(count_pos_test[6]+count_neg_test[6]))
                if count_pos_test[7] == 0:
                    self.exp.log_scalar("closeDW_p_test{}".format(iter_evl), count_pos_test[7])
                else:
                    self.exp.log_scalar("closeDW_p_test{}".format(iter_evl), count_pos_test[7]/(count_pos_test[7]+count_neg_test[7]))
                if count_pos_test[8] == 0:
                    self.exp.log_scalar("opendrawer1_p_test{}".format(iter_evl), count_pos_test[8])
                else:
                    self.exp.log_scalar("opendrawer1_p_test{}".format(iter_evl), count_pos_test[8]/(count_pos_test[8]+count_neg_test[8]))
                if count_pos_test[9] == 0:
                    self.exp.log_scalar("closedrawer1_p_test{}".format(iter_evl), count_pos_test[9])
                else:
                    self.exp.log_scalar("closedrawer1_p_test{}".format(iter_evl), count_pos_test[9]/(count_pos_test[9]+count_neg_test[9]))
                if count_pos_test[10] == 0:
                    self.exp.log_scalar("opendrawer2_p_test{}".format(iter_evl), count_pos_test[10])
                else:
                    self.exp.log_scalar("opendrawer2_p_test{}".format(iter_evl), count_pos_test[10]/(count_pos_test[10]+count_neg_test[10]))
                if count_pos_test[11] == 0:
                    self.exp.log_scalar("closedrawer2_p_test{}".format(iter_evl), count_pos_test[11])
                else:
                    self.exp.log_scalar("closedrawer2_p_test{}".format(iter_evl), count_pos_test[11]/(count_pos_test[11]+count_neg_test[11]))
                if count_pos_test[12] == 0:
                    self.exp.log_scalar("opendrawer3_p_test{}".format(iter_evl), count_pos_test[12])
                else:
                    self.exp.log_scalar("opendrawer3_p_test{}".format(iter_evl), count_pos_test[12]/(count_pos_test[12]+count_neg_test[12]))
                if count_pos_test[13] == 0:
                    self.exp.log_scalar("closedrawer3_p_test{}".format(iter_evl), count_pos_test[13])
                else:
                    self.exp.log_scalar("closedrawer3_p_test{}".format(iter_evl), count_pos_test[13]/(count_pos_test[13]+count_neg_test[13]))
                if count_pos_test[14] == 0:
                    self.exp.log_scalar("cleantable_p_test{}".format(iter_evl), count_pos_test[14])
                else:
                    self.exp.log_scalar("cleantable_p_test{}".format(iter_evl), count_pos_test[14]/(count_pos_test[14]+count_neg_test[14]))
                if count_pos_test[15] == 0:
                    self.exp.log_scalar("drinkcup_p_test{}".format(iter_evl), count_pos_test[15])
                else:
                    self.exp.log_scalar("drinkcup_p_test{}".format(iter_evl), count_pos_test[15]/(count_pos_test[15]+count_neg_test[15]))
                if count_pos_test[16] == 0:
                    self.exp.log_scalar("toggle_p_test{}".format(iter_evl), count_pos_test[16])
                else:
                    self.exp.log_scalar("toggle_p_test{}".format(iter_evl), count_pos_test[16]/(count_pos_test[16]+count_neg_test[16]))
                if count_pos_test[17] == 0:
                    self.exp.log_scalar("none_p_test{}".format(iter_evl), count_pos_test[17])
                else:
                    self.exp.log_scalar("none_p_test{}".format(iter_evl), count_pos_test[17]/(count_pos_test[17]+count_neg_test[17]))
                    
                if count_neg_test[0] == 0:
                    self.exp.log_scalar("opendoor1_n_test{}".format(iter_evl), count_neg_test[0])
                else:
                    self.exp.log_scalar("opendoor1_n_test{}".format(iter_evl), count_neg_test[0]/(count_pos_test[0]+count_neg_test[0]))
                if count_neg_test[1] == 0:
                    self.exp.log_scalar("opendoor2_n_test{}".format(iter_evl), count_neg_test[1])
                else:
                    self.exp.log_scalar("opendoor2_n_test{}".format(iter_evl), count_neg_test[1]/(count_pos_test[1]+count_neg_test[1]))
                if count_neg_test[2] == 0:
                    self.exp.log_scalar("closedoor1_n_test{}".format(iter_evl), count_neg_test[2])
                else:
                    self.exp.log_scalar("closedoor1_n_test{}".format(iter_evl), count_neg_test[2]/(count_pos_test[2]+count_neg_test[2]))
                if count_neg_test[3] == 0:
                    self.exp.log_scalar("closedoor2_n_test{}".format(iter_evl), count_neg_test[3])
                else:
                    self.exp.log_scalar("closedoor2_n_test{}".format(iter_evl), count_neg_test[3]/(count_pos_test[3]+count_neg_test[3]))
                if count_neg_test[4] == 0:
                    self.exp.log_scalar("openfridge_n_test{}".format(iter_evl), count_neg_test[4])
                else:
                    self.exp.log_scalar("openfridge_n_test{}".format(iter_evl), count_neg_test[4]/(count_pos_test[4]+count_neg_test[4]))
                if count_neg_test[5] == 0:
                    self.exp.log_scalar("closefridge_n_test{}".format(iter_evl), count_neg_test[5])
                else:
                    self.exp.log_scalar("closefridge_n_test{}".format(iter_evl), count_neg_test[5]/(count_pos_test[5]+count_neg_test[5]))
                if count_neg_test[6] == 0:
                    self.exp.log_scalar("openDW_n_test{}".format(iter_evl), count_neg_test[6])
                else:
                    self.exp.log_scalar("openDW_n_test{}".format(iter_evl), count_neg_test[6]/(count_pos_test[6]+count_neg_test[6]))
                if count_neg_test[7] == 0:
                    self.exp.log_scalar("closeDW_n_test{}".format(iter_evl), count_neg_test[7])
                else:
                    self.exp.log_scalar("closeDW_n_test{}".format(iter_evl), count_neg_test[7]/(count_pos_test[7]+count_neg_test[7]))
                if count_neg_test[8] == 0:
                    self.exp.log_scalar("opendrawer1_n_test{}".format(iter_evl), count_neg_test[8])
                else:
                    self.exp.log_scalar("opendrawer1_n_test{}".format(iter_evl), count_neg_test[8]/(count_pos_test[8]+count_neg_test[8]))
                if count_neg_test[9] == 0:
                    self.exp.log_scalar("closedrawer1_n_test{}".format(iter_evl), count_neg_test[9])
                else:
                    self.exp.log_scalar("closedrawer1_n_test{}".format(iter_evl), count_neg_test[9]/(count_pos_test[9]+count_neg_test[9]))
                if count_neg_test[10] == 0:
                    self.exp.log_scalar("opendrawer2_n_test{}".format(iter_evl), count_neg_test[10])
                else:
                    self.exp.log_scalar("opendrawer2_n_test{}".format(iter_evl), count_neg_test[10]/(count_pos_test[10]+count_neg_test[10]))
                if count_neg_test[11] == 0:
                    self.exp.log_scalar("closedrawer2_n_test{}".format(iter_evl), count_neg_test[11])
                else:
                    self.exp.log_scalar("closedrawer2_n_test{}".format(iter_evl), count_neg_test[11]/(count_pos_test[11]+count_neg_test[11]))
                if count_neg_test[12] == 0:
                    self.exp.log_scalar("opendrawer3_n_test{}".format(iter_evl), count_neg_test[12])
                else:
                    self.exp.log_scalar("opendrawer3_n_test{}".format(iter_evl), count_neg_test[12]/(count_pos_test[12]+count_neg_test[12]))
                if count_neg_test[13] == 0:
                    self.exp.log_scalar("closedrawer3_n_test{}".format(iter_evl), count_neg_test[13])
                else:
                    self.exp.log_scalar("closedrawer3_n_test{}".format(iter_evl), count_neg_test[13]/(count_pos_test[13]+count_neg_test[13]))
                if count_neg_test[14] == 0:
                    self.exp.log_scalar("cleantable_n_test{}".format(iter_evl), count_neg_test[14])
                else:
                    self.exp.log_scalar("cleantable_n_test{}".format(iter_evl), count_neg_test[14]/(count_pos_test[14]+count_neg_test[14]))
                if count_neg_test[15] == 0:
                    self.exp.log_scalar("drinkcup_n_test{}".format(iter_evl), count_neg_test[15])
                else:
                    self.exp.log_scalar("drinkcup_n_test{}".format(iter_evl), count_neg_test[15]/(count_pos_test[15]+count_neg_test[15]))
                if count_neg_test[16] == 0:
                    self.exp.log_scalar("toggle_n_test{}".format(iter_evl), count_neg_test[16])
                else:
                    self.exp.log_scalar("toggle_n_test{}".format(iter_evl), count_neg_test[16]/(count_pos_test[16]+count_neg_test[16]))
                if count_neg_test[17] == 0:
                    self.exp.log_scalar("none_n_test{}".format(iter_evl), count_neg_test[17])
                else:
                    self.exp.log_scalar("none_n_test{}".format(iter_evl), c_neg_val[17]/(count_pos_test[17]+count_neg_test[17]))
                '''

        if self.config["usage_modus"] == "train":
            logging.info('    Network_selecter:    Train:    eliminating network file')
            os.remove(self.config['folder_exp'] + 'network.pt')
        
            
        torch.cuda.empty_cache()
        del count_neg_test,count_pos_test
        del results_test
        del c_neg_val, c_pos_val
        del results_train
        del acc_train_ac
        del f1_weighted_train_ac 
        del f1_mean_train_ac
        del precisions_test
        del recalls_test 
        del acc_test_ac 
        del f1_weighted_test_ac
        del f1_mean_test_ac
        
        if self.config['output']== 'attribute':
            del p
            del precisions_attr_test
            del recalls_attr_test
            del acc_attr_test_ac 

        return


    def test(self, testing = False):
        """
        Test method. Testing the network , saving the performances

        @param testing: Enabling testing after training
        @return results_test: dict with the results of the testing
        @return confusion_matrix_test: confusion matrix of the text
        """

        start_time_test = time.time()
        precisions_test = []
        recalls_test = []
        if self.config['output']== 'attribute':
            precisions_attr_test = []
            recalls_attr_test = []

        # Testing the network in folder (according to the conf)
        results_test, confusion_matrix_test, _ , c_pos_test, c_neg_test = self.network.evolution_evaluation(ea_iter=0, testing=testing)
        #results_test, confusion_matrix_test, _ = self.network.evolution_evaluation(ea_iter=0, testing=testing)

        elapsed_time_test = time.time() - start_time_test

        # Appending results for later saving in results file
        precisions_test.append(results_test['precision'].numpy())
        recalls_test.append(results_test['recall'].numpy())
        if self.config['output']== 'attribute':
            precisions_attr_test.append(results_test['precision_attr'].numpy())
            recalls_attr_test.append(results_test['recall_attr'].numpy())
        
        if self.config['output']== 'softmax':
                logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                     'f1_weighted {}, f1_mean {}'.format(elapsed_time_test, results_test['acc'],
                                                         results_test['f1_weighted'], results_test['f1_mean']))
        elif self.config['output']== 'attribute':
                logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                     'f1_weighted {}, f1_mean {}, acc_attr {}'.format(elapsed_time_test, results_test['acc'],
                                                         results_test['f1_weighted'], results_test['f1_mean'], results_test['acc_attrs']))

        # Saving the results
        if not testing:
            if self.config['output']== 'softmax':
                self.save([results_test['acc']], [results_test['f1_weighted']], [results_test['f1_mean']],
                      type_simple='testing', confusion_matrix=confusion_matrix_test,
                      time_iter=elapsed_time_test, precisions=np.array(precisions_test),
                      recalls=np.array(recalls_test))
            elif self.config['output']== 'attribute':
                self.save([results_test['acc']], [results_test['f1_weighted']], [results_test['f1_mean']],
                      type_simple='testing', confusion_matrix=confusion_matrix_test,
                      time_iter=elapsed_time_test, precisions=np.array(precisions_test),
                      recalls=np.array(recalls_test), acc_attr_test=[results_test['acc_attrs']], precisions_attr=np.array(precisions_attr_test),
                      recalls_attr=np.array(recalls_attr_test)  )
            return
        
        torch.cuda.empty_cache()
        del precisions_test 
        del recalls_test 
        if self.config['output']== 'attribute':
            del precisions_attr_test 
            del recalls_attr_test 
        
        return results_test, confusion_matrix_test, c_pos_test, c_neg_test
        #return results_test, confusion_matrix_test

    def net_modus(self):
        """
        Setting the training, validation, evolution and final training.
        """
        logging.info('    Network_selecter: Net modus: {}'.format(self.config['usage_modus']))
        if self.config['usage_modus'] == 'train':
            self.train(itera=1, testing=True)
        elif self.config['usage_modus'] == 'test':
            self.test()
        #elif self.config['usage_modus'] == 'evolution':
            # Not implementing here, see paper ICPR2018
            #self.evolution()
        elif self.config['usage_modus'] == 'train_final':
            self.train(itera=1,  testing=True)
        elif self.config['usage_modus'] == 'fine_tuning':
            self.train(itera=5, testing=True)
        return
