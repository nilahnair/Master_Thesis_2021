'''
Created on May 17, 2019

@author: fmoya
'''

from __future__ import print_function
import logging
import os

from network_user_act import Network_User
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
            results_train, confusion_matrix_train, best_itera, c_pos_val, c_neg_val, hidden, cell = self.network.evolution_evaluation(ea_iter=iter_evl)
            
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
                self.exp.log_scalar("standing_pos_val_final{}".format(iter_evl), c_pos_val[0])
            else:
                self.exp.log_scalar("standing_pos_val_final{}".format(iter_evl), c_pos_val[0]/(c_pos_val[0]+c_neg_val[0]))
            if c_pos_val[1] == 0:
                self.exp.log_scalar("walking_pos_val_final{}".format(iter_evl), c_pos_val[1])
            else:
                self.exp.log_scalar("walking_pos_val_final{}".format(iter_evl), c_pos_val[1]/(c_pos_val[1]+c_neg_val[1]))
            if c_pos_val[2] == 0:
                self.exp.log_scalar("cart_pos_val_final{}".format(iter_evl), c_pos_val[2])
            else:
                self.exp.log_scalar("cart_pos_val_final{}".format(iter_evl), c_pos_val[2]/(c_pos_val[2]+c_neg_val[2]))
            if c_pos_val[3] == 0:
                self.exp.log_scalar("handling_up_pos_val_final{}".format(iter_evl), c_pos_val[3])
            else:
                self.exp.log_scalar("handling_up_pos_val_final{}".format(iter_evl), c_pos_val[3]/(c_pos_val[3]+c_neg_val[3]))
            if c_pos_val[4] == 0:
                self.exp.log_scalar("handling_cen_pos_val_final{}".format(iter_evl), c_pos_val[4])
            else:
                self.exp.log_scalar("handling_cen_pos_val_final{}".format(iter_evl), c_pos_val[4]/(c_pos_val[4]+c_neg_val[4]))
            if c_pos_val[5] == 0:
                self.exp.log_scalar("handling_down_pos_val_final{}".format(iter_evl), c_pos_val[5])
            else:
                self.exp.log_scalar("handling_down_pos_val_final{}".format(iter_evl), c_pos_val[5]/(c_pos_val[5]+c_neg_val[5]))
            if c_pos_val[6] == 0:
                self.exp.log_scalar("synch_pos_val_final{}".format(iter_evl), c_pos_val[6])
            else:
                self.exp.log_scalar("synch_pos_val_final{}".format(iter_evl), c_pos_val[6]/(c_pos_val[6]+c_neg_val[6]))
            if c_pos_val[7] == 0:
                self.exp.log_scalar("none_pos_val_final{}".format(iter_evl), c_pos_val[7])
            else:
                self.exp.log_scalar("none_pos_val_final{}".format(iter_evl), c_pos_val[7]/(c_pos_val[7]+c_neg_val[7]))
            
            if c_neg_val[0] == 0:
                self.exp.log_scalar("standing_neg_val_final{}".format(iter_evl), c_neg_val[0])
            else:
                self.exp.log_scalar("standing_neg_val_final{}".format(iter_evl), c_neg_val[0]/(c_pos_val[0]+c_neg_val[0]))
            if c_neg_val[1] == 0:
                self.exp.log_scalar("walking_neg_val_final{}".format(iter_evl), c_neg_val[1])
            else:
                self.exp.log_scalar("walking_neg_val_final{}".format(iter_evl), c_neg_val[1]/(c_pos_val[1]+c_neg_val[1]))
            if c_neg_val[2] == 0:
                self.exp.log_scalar("cart_neg_val_final{}".format(iter_evl), c_neg_val[2])
            else:
                self.exp.log_scalar("cart_neg_val_final{}".format(iter_evl), c_neg_val[2]/(c_pos_val[2]+c_neg_val[2]))
            if c_neg_val[3] == 0:
                self.exp.log_scalar("handling_up_neg_val_final{}".format(iter_evl), c_neg_val[3])
            else:
                self.exp.log_scalar("handling_up_neg_val_final{}".format(iter_evl), c_neg_val[3]/(c_pos_val[3]+c_neg_val[3]))
            if c_neg_val[4] == 0:
                self.exp.log_scalar("handling_cen_neg_val_final{}".format(iter_evl), c_neg_val[4])
            else:
                self.exp.log_scalar("handling_cen_neg_val_final{}".format(iter_evl), c_neg_val[4]/(c_pos_val[4]+c_neg_val[4]))
            if c_neg_val[5] == 0:
                self.exp.log_scalar("handling_down_neg_val_final{}".format(iter_evl), c_neg_val[5])
            else:
                self.exp.log_scalar("handling_down_neg_val_final{}".format(iter_evl), c_neg_val[5]/(c_pos_val[5]+c_neg_val[5]))
            if c_neg_val[6] == 0:
                self.exp.log_scalar("synch_neg_val_final{}".format(iter_evl), c_neg_val[6])
            else:
                self.exp.log_scalar("synch_neg_val_final{}".format(iter_evl), c_neg_val[6]/(c_pos_val[6]+c_neg_val[6]))
            if c_neg_val[7] == 0:
                self.exp.log_scalar("none_neg_val_final{}".format(iter_evl), c_neg_val[7])
            else:
                self.exp.log_scalar("none_neg_val_final{}".format(iter_evl), c_neg_val[7]/(c_pos_val[7]+c_neg_val[7]))
                                      
            # Saving the results
            self.save(acc_train_ac, f1_weighted_train_ac, f1_mean_train_ac, time_iter=time_train, precisions=results_train['precision'], 
                      recalls=results_train['recall'], best_itera=best_itera)
            
            # Testing the network
            if testing:
                start_time_test = time.time()
                results_test, confusion_matrix_test, count_pos_test, count_neg_test = self.test(testing=True, hidden= hidden, cell=cell)
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
                self.exp.log_scalar("standing_pos_test{}".format(iter_evl), count_pos_test[0])
            else:
                self.exp.log_scalar("standing_pos_test{}".format(iter_evl), count_pos_test[0]/(count_pos_test[0]+count_neg_test[0]))
            if count_pos_test[1] == 0:
                self.exp.log_scalar("walking_pos_test{}".format(iter_evl), count_pos_test[1])
            else:
                self.exp.log_scalar("walking_pos_test{}".format(iter_evl), count_pos_test[1]/(count_pos_test[1]+count_neg_test[1]))
            if count_pos_test[2] == 0:
                self.exp.log_scalar("cart_pos_test{}".format(iter_evl), count_pos_test[2])
            else:
                self.exp.log_scalar("cart_pos_test{}".format(iter_evl), count_pos_test[2]/(count_pos_test[2]+count_neg_test[2]))
            if count_pos_test[3] == 0:
                self.exp.log_scalar("handling_up_pos_test{}".format(iter_evl), count_pos_test[3])
            else:
                self.exp.log_scalar("handling_up_pos_test{}".format(iter_evl), count_pos_test[3]/(count_pos_test[3]+count_neg_test[3]))
            if count_pos_test[4] == 0:
                self.exp.log_scalar("handling_cen_pos_test{}".format(iter_evl), count_pos_test[4])
            else:
                self.exp.log_scalar("handling_cen_pos_test{}".format(iter_evl), count_pos_test[4]/(count_pos_test[4]+count_neg_test[4]))
            if count_pos_test[5] == 0:
                self.exp.log_scalar("handling_down_pos_test{}".format(iter_evl), count_pos_test[5])
            else:
                self.exp.log_scalar("handling_down_pos_test{}".format(iter_evl), count_pos_test[5]/(count_pos_test[5]+count_neg_test[5]))
            if count_pos_test[6] == 0:
                self.exp.log_scalar("synch_pos_test{}".format(iter_evl), count_pos_test[6])
            else:
                self.exp.log_scalar("synch_pos_test{}".format(iter_evl), count_pos_test[6]/(count_pos_test[6]+count_neg_test[6]))
            if count_pos_test[7] == 0:
                self.exp.log_scalar("none_pos_test{}".format(iter_evl), count_pos_test[7])
            else:
                self.exp.log_scalar("none_pos_test{}".format(iter_evl), count_pos_test[7]/(count_pos_test[7]+count_neg_test[7]))
                
            self.exp.log_scalar("total_standing_pos_test{}".format(iter_evl), count_pos_test[0])
            self.exp.log_scalar("total_walking_pos_test{}".format(iter_evl), count_pos_test[1])
            self.exp.log_scalar("total_cart_pos_test{}".format(iter_evl), count_pos_test[2])
            self.exp.log_scalar("total_handling_up_pos_test{}".format(iter_evl), count_pos_test[3])
            self.exp.log_scalar("total_handling_cen_pos_test{}".format(iter_evl), count_pos_test[4])
            self.exp.log_scalar("total_handling_down_pos_test{}".format(iter_evl), count_pos_test[5])
            self.exp.log_scalar("total_synch_pos_test{}".format(iter_evl), count_pos_test[6])
            
            if count_neg_test[0] == 0:
                self.exp.log_scalar("standing_neg_test{}".format(iter_evl), count_neg_test[0])
            else:
                self.exp.log_scalar("standing_neg_test{}".format(iter_evl), count_neg_test[0]/(count_pos_test[0]+count_neg_test[0]))
            if count_neg_test[1] == 0:
                self.exp.log_scalar("walking_neg_test{}".format(iter_evl), count_neg_test[1])
            else:
                self.exp.log_scalar("walking_neg_test{}".format(iter_evl), count_neg_test[1]/(count_pos_test[1]+count_neg_test[1]))
            if count_neg_test[2] == 0:
                self.exp.log_scalar("cart_neg_test{}".format(iter_evl), count_neg_test[2])
            else:
                self.exp.log_scalar("cart_neg_test{}".format(iter_evl), count_neg_test[2]/(count_pos_test[2]+count_neg_test[2]))
            if count_neg_test[3] == 0:
                self.exp.log_scalar("handling_up_neg_test{}".format(iter_evl), count_neg_test[3])
            else:
                self.exp.log_scalar("handling_up_neg_test{}".format(iter_evl), count_neg_test[3]/(count_pos_test[3]+count_neg_test[3]))
            if count_neg_test[4] == 0:
                self.exp.log_scalar("shandling_cen_neg_test{}".format(iter_evl), count_neg_test[4])
            else:
                self.exp.log_scalar("handling_cen_neg_test{}".format(iter_evl), count_neg_test[4]/(count_pos_test[4]+count_neg_test[4]))
            if count_neg_test[5] == 0:
                self.exp.log_scalar("handling_down_neg_test{}".format(iter_evl), count_neg_test[5])
            else:
                self.exp.log_scalar("handling_down_neg_test{}".format(iter_evl), count_neg_test[5]/(count_pos_test[5]+count_neg_test[5]))
            if count_neg_test[6] == 0:
                self.exp.log_scalar("synch_neg_test{}".format(iter_evl), count_neg_test[6])
            else:
                self.exp.log_scalar("synch_neg_test{}".format(iter_evl), count_neg_test[6]/(count_pos_test[6]+count_neg_test[6]))
            if count_neg_test[7] == 0:
                self.exp.log_scalar("none_neg_test{}".format(iter_evl), count_neg_test[7])
            else:
                self.exp.log_scalar("none_neg_test{}".format(iter_evl), count_neg_test[7]/(count_pos_test[7]+count_neg_test[7]))
            

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


    def test(self, testing = False, hidden=0, cell=0):
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
        results_test, confusion_matrix_test, _ , c_pos_test, c_neg_test = self.network.evolution_evaluation(ea_iter=0, testing=testing, hidden=hidden, cell=cell)

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
