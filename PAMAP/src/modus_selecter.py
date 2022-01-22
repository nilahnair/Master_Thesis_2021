'''
Created on Feb 27, 2019

@author: fmoya

Old network_selecter with caffe/theano implementations

'''

from __future__ import print_function
import logging

from network_user import Network_User
from attributes import Attributes

import time

import xml.etree.ElementTree as ET
from xml.dom import minidom

import sys
import os

import numpy as np

from sacred import Experiment


class Modus_Selecter(object):
    '''
    classdocs
    '''

    def __init__(self, config, exp=None):
        '''
        Constructor
        '''

        logging.info('    Network_selecter: Constructor')
        self.config = config

        self.exp = exp
        self.network = Network_User(config, self.exp)
        self.attributes = Attributes(config)
        self.attrs_0 = None

        self.xml_root = ET.Element("Experiment_{}".format(self.config["name_counter"]))

        return

    def save(self, acc_test, f1_weighted_test, f1_mean_test, ea_iter, type_simple='training', confusion_matrix=0,
             time_iter=0, precisions=0, recalls=0, best_itera=0, acc_test_seg=[0], f1_weighted_test_seg=[0],
             f1_mean_test_seg=[0]):
        """
        Save the results of traiing and testing according to the configuration.
        As training is repeated several times, results are appended, and mean and std of all the repetitions
        are computed.

        @param acc_test: List of accuracies of val or testing
        @param f1_weighted_test: List of F1w of val or testing
        @param f1_mean_test: List of F1m of val or testing
        @param ea_iter: Iteration of evolution
        @param type_simple: Type of experiment
        @param confusion_matrix: Confusion Matrix
        @param time_iter: Time of experiment run
        @param precisions: List of class precisions
        @param recalls: List of class recalls
        @param best_itera: Best evolution iteration
        """

        child_network = ET.SubElement(self.xml_root, "network", network=str(self.config['network']))
        child_dataset = ET.SubElement(child_network, "dataset", dataset=str(self.config['dataset']))
        child = ET.SubElement(child_dataset, "usage_modus", usage_modus=str(self.config['usage_modus']))
        child = ET.SubElement(child_dataset, "dataset_finetuning",
                              dataset_finetuning=str(self.config['dataset_finetuning']))
        child = ET.SubElement(child_dataset, "percentages_names",
                              proportions=str(self.config['proportions']))
        child = ET.SubElement(child_dataset, "type_simple", type_simple=str(type_simple))
        child = ET.SubElement(child_dataset, "output", output=str(self.config['output']))
        child = ET.SubElement(child_dataset, "pooling", pooling=str(self.config['pooling']))

        child = ET.SubElement(child_dataset, "lr", lr=str(self.config['lr']))
        child = ET.SubElement(child_dataset, "epochs", epochs=str(self.config['epochs']))
        child = ET.SubElement(child_dataset, "distance", distance=str(self.config['distance']))
        child = ET.SubElement(child_dataset, "reshape_input", reshape_input=str(self.config["reshape_input"]))

        child = ET.SubElement(child_dataset, "ea_iter", ea_iter=str(ea_iter))
        child = ET.SubElement(child_dataset, "freeze_options", freeze_options=str(self.config['freeze_options']))
        child = ET.SubElement(child_dataset, "time_iter", time_iter=str(time_iter))
        child = ET.SubElement(child_dataset, "best_itera", best_itera=str(best_itera))

        for exp in range(len(acc_test)):
            child = ET.SubElement(child_dataset, "metrics", acc_test=str(acc_test[exp]),
                                  f1_weighted_test=str(f1_weighted_test[exp]),
                                  f1_mean_test=str(f1_mean_test[exp]))
        child = ET.SubElement(child_dataset, "metrics_mean", acc_test_mean=str(np.mean(acc_test)),
                              f1_weighted_test_mean=str(np.mean(f1_weighted_test)),
                              f1_mean_test_mean=str(np.mean(f1_mean_test)))
        child = ET.SubElement(child_dataset, "metrics_std", acc_test_std=str(np.std(acc_test)),
                              f1_weighted_test_std=str(np.std(f1_weighted_test)),
                              f1_mean_test_std=str(np.std(f1_mean_test)))
        child = ET.SubElement(child_dataset, "confusion_matrix_last",
                              confusion_matrix_last=str(confusion_matrix))
        if type_simple == 'training':
            child = ET.SubElement(child_dataset, "precision", precision=str(precisions))
            child = ET.SubElement(child_dataset, "recall", recall=str(recalls))
        else:
            child = ET.SubElement(child_dataset, "precision_mean", precision_mean=str(np.mean(precisions, axis=0)))
            child = ET.SubElement(child_dataset, "precision_std", precision_std=str(np.std(precisions, axis=0)))
            child = ET.SubElement(child_dataset, "recall_mean", recall_mean=str(np.mean(recalls, axis=0)))
            child = ET.SubElement(child_dataset, "recall_std", recall_std=str(np.std(recalls, axis=0)))

        for exp in range(len(acc_test_seg)):
            child = ET.SubElement(child_dataset, "metrics_seg", acc_test_seg=str(acc_test_seg[exp]),
                                  f1_weighted_test_seg=str(f1_weighted_test_seg[exp]),
                                  f1_mean_test_seg=str(f1_mean_test_seg[exp]))
        child = ET.SubElement(child_dataset, "metrics_seg_mean", acc_test_seg_mean=str(np.mean(acc_test_seg)),
                              f1_weighted_test_seg_mean=str(np.mean(f1_weighted_test_seg)),
                              f1_mean_test_seg_mean=str(np.mean(f1_mean_test_seg)))
        child = ET.SubElement(child_dataset, "metrics_seg_std", acc_test_seg_std=str(np.std(acc_test_seg)),
                              f1_weighted_test_seg_std=str(np.std(f1_weighted_test_seg)),
                              f1_mean_test_seg_std=str(np.std(f1_mean_test_seg)))

        return

    def train(self, itera=1, testing=False):
        """
        Train method. Train network for a certain number of repetitions
        computing the val performance, Testing using test(), saving the performances

        @param itera: training iteration, as training is repeated X number of times
        @param testing: Enabling testing after training
        """

        global f1_weighted_test_ac
        logging.info('    Network_selecter: Train')

        acc_train_ac = []
        f1_weighted_train_ac = []
        f1_mean_train_ac = []
        precisions_test = []
        recalls_test = []
        acc_train_seg_ac = []
        f1_weighted_train_seg_ac = []
        f1_mean_train_seg_ac = []

        if testing:
            acc_test_ac = []
            f1_weighted_test_ac = []
            f1_mean_test_ac = []
            acc_test_seg_ac = []
            f1_weighted_test_seg_ac = []
            f1_mean_test_seg_ac = []

        #There will be only one iteration
        #As there is not evolution
        for iter_evl in range(itera):
            start_time_train = time.time()
            logging.info('    Network_selecter:    Train iter {}'.format(iter_evl))
            # Training the network and obtaining the validation results
            #acc_train, f1_weighted_train, f1_mean_train, _ = self.network.evolution_evaluation(ea_iter=iter_evl)
            results_train, confusion_matrix_train, best_itera = self.network.evolution_evaluation(ea_iter=iter_evl)

            # Appending results for later saving in results file
            acc_train_ac.append(results_train["classification"]['acc'])
            f1_weighted_train_ac.append(results_train["classification"]['f1_weighted'])
            f1_mean_train_ac.append(results_train["classification"]['f1_mean'])
            acc_train_seg_ac.append(results_train["segmentation"]['acc'])
            f1_weighted_train_seg_ac.append(results_train["segmentation"]['f1_weighted'])
            f1_mean_train_seg_ac.append(results_train["segmentation"]['f1_mean'])

            time_train = time.time() - start_time_train

            logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                         'f1_weighted {}, f1_mean {}'.format(time_train,
                                                             results_train["classification"]['acc'],
                                                             results_train["classification"]['f1_weighted'],
                                                             results_train["classification"]['f1_mean']))
            # Saving the results
            if self.config["sacred"]:
                self.exp.log_scalar("Acc_Val", value=results_train["classification"]['acc'])
                self.exp.log_scalar("F1w_Val", value=results_train["classification"]['f1_weighted'])
                self.exp.log_scalar("F1m_Val", value=results_train["classification"]['f1_mean'])
                self.exp.log_scalar("Acc_Seg_Val", value=results_train["segmentation"]['acc'])
                self.exp.log_scalar("F1w_Seg_Val", value=results_train["segmentation"]['f1_weighted'])
                self.exp.log_scalar("F1m_Seg_Val", value=results_train["segmentation"]['f1_mean'])
                self.exp.log_scalar("best_itera", value=best_itera)

            self.save(acc_train_ac, f1_weighted_train_ac, f1_mean_train_ac, ea_iter=iter_evl,
                      time_iter=time_train, precisions=results_train["classification"]['precision'],
                      recalls=results_train["classification"]['recall'], best_itera=best_itera,
                      acc_test_seg=acc_train_seg_ac, f1_weighted_test_seg=f1_weighted_train_seg_ac,
                      f1_mean_test_seg=f1_mean_train_seg_ac)

            # Testing the network
            if testing:
                start_time_test = time.time()
                results_test, confusion_matrix_test = self.test(testing=True)
                acc_test_ac.append(results_test["classification"]['acc'])
                f1_weighted_test_ac.append(results_test["classification"]['f1_weighted'])
                f1_mean_test_ac.append(results_test["classification"]['f1_mean'])
                acc_test_seg_ac.append(results_test["segmentation"]['acc'])
                f1_weighted_test_seg_ac.append(results_test["segmentation"]['f1_weighted'])
                f1_mean_test_seg_ac.append(results_test["segmentation"]['f1_mean'])
                precisions_test.append(results_test["classification"]['precision'].numpy())
                recalls_test.append(results_test["classification"]['recall'].numpy())

                time_test = time.time() - start_time_test

                if self.config["sacred"]:
                    self.exp.log_scalar("Acc_Test", value=results_test["classification"]['acc'])
                    self.exp.log_scalar("F1w_Test", value=results_test["classification"]['f1_weighted'])
                    self.exp.log_scalar("F1m_Test", value=results_test["classification"]['f1_mean'])
                    self.exp.log_scalar("Acc_Seg_Test", value=results_test["segmentation"]['acc'])
                    self.exp.log_scalar("F1w_Seg_Test", value=results_test["segmentation"]['f1_weighted'])
                    self.exp.log_scalar("F1m_Seg_Test", value=results_test["segmentation"]['f1_mean'])
                    self.save(acc_test_ac, f1_weighted_test_ac, f1_mean_test_ac, ea_iter=iter_evl,
                              type_simple='testing',
                              confusion_matrix=confusion_matrix_test, time_iter=time_test,
                              precisions=np.array(precisions_test),
                              recalls=np.array(recalls_test),
                              acc_test_seg=acc_test_seg_ac,
                              f1_weighted_test_seg=f1_weighted_test_seg_ac,
                              f1_mean_test_seg=f1_mean_test_seg_ac
                    )

            self.network.restart_network()

        if testing:
            self.save(acc_test_ac, f1_weighted_test_ac, f1_mean_test_ac, ea_iter=iter_evl, type_simple='testing',
                      confusion_matrix=confusion_matrix_test, time_iter=time_test, precisions=np.array(precisions_test),
                      recalls=np.array(recalls_test),
                      acc_test_seg=acc_test_seg_ac,
                      f1_weighted_test_seg=f1_weighted_test_seg_ac,
                      f1_mean_test_seg=f1_mean_test_seg_ac)

        if self.config["usage_modus"] == "train":
            logging.info('    Network_selecter:    Train:    eliminating network file')
            os.remove(self.config['folder_exp'] + 'network.pt')

        return

    def test(self, testing = False):
        """
        Test method. Testing the network , saving the performances

        @param itera: training iteration, as training is repeated X number of times
        @param testing: Enabling testing after training
        @return results_test: dict with the results of the testing
        @return confusion_matrix_test: confusion matrix of the text
        """

        start_time_test = time.time()
        precisions_test = []
        recalls_test = []

        # Testing the network in folder (according to the conf)
        results_test, confusion_matrix_test, _ = self.network.evolution_evaluation(ea_iter=0, testing=testing)

        elapsed_time_test = time.time() - start_time_test

        # Appending results for later saving in results file
        precisions_test.append(results_test["classification"]['precision'].numpy())
        recalls_test.append(results_test["classification"]['recall'].numpy())

        logging.info('    Network_selecter:    Train: elapsed time {} acc {}, '
                     'f1_weighted {}, f1_mean {}'.format(elapsed_time_test, results_test["classification"]['acc'],
                                                         results_test["classification"]['f1_weighted'],
                                                         results_test["classification"]['f1_mean']))

        # Saving the results
        if not testing:
            if self.config["sacred"]:
                self.exp.log_scalar("Acc_Test", value=results_test["classification"]['acc'])
                self.exp.log_scalar("F1w_Test", value=results_test["classification"]['f1_weighted'])
                self.exp.log_scalar("F1m_Test", value=results_test["classification"]['f1_mean'])
                self.exp.log_scalar("Acc_Seg_Test", value=results_test["segmentation"]['acc'])
                self.exp.log_scalar("F1w_Seg_Test", value=results_test["segmentation"]['f1_weighted'])
                self.exp.log_scalar("F1m_Seg_Test", value=results_test["segmentation"]['f1_mean'])

            self.save([results_test["classification"]['acc']], [results_test["classification"]['f1_weighted']],
                      [results_test["classification"]['f1_mean']],
                      ea_iter=0, type_simple='testing', confusion_matrix=confusion_matrix_test,
                      time_iter=elapsed_time_test, precisions=np.array(precisions_test),
                      recalls=np.array(recalls_test),
                      acc_test_seg=[results_test["segmentation"]['acc']],
                      f1_weighted_test_seg=[results_test["segmentation"]['acc']],
                      f1_mean_test_seg=[results_test["segmentation"]['acc']]
                      )
            return

        return results_test, confusion_matrix_test


    def evolution(self):
        logging.info('    Network_selecter: Evolution')

        # Setting attribute population
        if os.path.isfile('../' + self.config['folder_exp'] + '/iters.txt'):
            best_attrs = self.attributes.load_attrs(0, name_file='best_attrs')
            self.attrs_0 = best_attrs[0]['attrs']
            self.network.set_attrs(self.attrs_0)
            init_iter = self.load_iters() + 1

            logging.info('    Network_selecter:     Loading previous training in iters {}...'.format(init_iter))
        else:
            self.attrs_0 = self.attributes.creating_init_population()
            init_iter = 0
            self.network.set_attrs(self.attrs_0)

            logging.info('    Network_selecter:     No Loading training in iters {}...'.format(init_iter))

        start_time_test = time.time()

        # initial evaluation of the population number 0
        acc_test, f1_weighted_test, f1_mean_test = self.network.evolution_evaluation(ea_iter=0)

        elapsed_time_test = time.time() - start_time_test

        logging.info(
            '    Network_selecter:     EA: elapsed time {} acc {}, f1_weighted {}, f1_mean {}'.format(elapsed_time_test,
                                                                                                      acc_test,
                                                                                                      f1_weighted_test,
                                                                                                      f1_mean_test))
        #Save validation results
        self.save(acc_test, f1_weighted_test, f1_mean_test, ea_iter = 0)

        self.attributes.save_attrs(self.attrs_0, f1_weighted_test, init_iter, name_file='attrs')

        #Setting up the fitness
        best_fitness = f1_weighted_test
        best_attr = np.copy(self.attrs_0)

        fitness = []
        all_fitness = []
        all_acc = []
        iters = []

        fitness.append(f1_weighted_test)
        all_fitness.append(f1_weighted_test)
        all_acc.append(acc_test)
        iters.append(init_iter)

        np.savetxt(self.config["folder_exp"] + 'fitness.txt', fitness, fmt='%.5f')
        np.savetxt(self.config["folder_exp"] + 'iters.txt', iters, fmt='%d')
        np.savetxt(self.config["folder_exp"] + 'best_attributes.txt', best_attr, fmt='%d')
        np.savetxt(self.config["folder_exp"] + 'all_fitness.txt', all_fitness, fmt='%.5f')
        np.savetxt(self.config["folder_exp"] + 'all_accuracies.txt', all_acc, fmt='%.5f')


        # Starting the evolution
        epochs_training = self.config["epochs"]
        for ea_iter in range(1, self.config["evolution_iter"]):

            logging.info(
                '    Network_selecter:     EA: iter {} from {} with epochs {}...'.format(ea_iter,
                                                                                         self.config["evolution_iter"],
                                                                                         epochs_training))
            #Mutating the attributes
            # attr_new = self.mutation_nonlocal_percentage(best_attr, best_percentage, number_K = 8)
            # attr_new = self.mutation_local(best_attr)
            # attr_new = self.mutation_nonlocal(best_attr, number_K = 4)
            attr_new = self.attributes.mutation_global(best_attr)

            #Setting the new attributes to the network
            self.network.set_attrs(attr_new)

            #training and validating the network
            acc_test, f1_weighted_test, f1_mean_test = self.network.evolution_evaluation(ea_iter=ea_iter)

            logging.info('    Network_selecter:     EA: elapsed time {} acc {}, f1_weighted {}, f1_mean {}'.format(
                elapsed_time_test,
                acc_test,
                f1_weighted_test,
                f1_mean_test))

            #Store the fitness
            all_fitness.append(f1_weighted_test)
            np.savetxt(self.config["folder_exp"] + 'all_fitness.txt', all_fitness, fmt='%.5f')

            self.save(acc_test, f1_weighted_test, f1_mean_test, ea_iter=ea_iter)

            all_acc.append(acc_test)
            np.savetxt(self.config["folder_exp"] + 'all_accuracies.txt', all_acc, fmt='%.5f')

            #save the attributes
            self.attributes.save_attrs(attr_new, f1_weighted_test, ea_iter, protocol_file='ab')

            #select if fitness improved, if so, update the fitness and save the network and attributes
            if f1_weighted_test > best_fitness:
                logging.info('    Network_selecter:     EA: Got best attrs with f1{}...'.format(f1_weighted_test))

                best_fitness = f1_weighted_test
                best_attr = np.copy(attr_new)

                fitness.append(f1_weighted_test)
                iters.append(ea_iter)

                #Saving the best attributes and its network
                self.attributes.save_attrs(attr_new, f1_weighted_test, ea_iter, name_file='best_attrs')
                self.network.save_network(ea_iter)

                np.savetxt(self.config["folder_exp"] + 'fitness.txt', fitness, fmt='%.5f')
                np.savetxt(self.config["folder_exp"] + 'iters.txt', iters, fmt='%d')
                np.savetxt(self.config["folder_exp"] + 'best_attributes.txt', best_attr, fmt='%d')

        return



    def net_modus(self):
        """
        Setting the training, validation, evolution and final training.
        """
        logging.info('    Network_selecter: Net modus: {}'.format(self.config['usage_modus']))
        if self.config['usage_modus'] == 'train':
            self.train(itera=5, testing=True)
        elif self.config['usage_modus'] == 'test':
            self.test()
        elif self.config['usage_modus'] == 'evolution':
            self.evolution()
        elif self.config['usage_modus'] == 'train_final':
            self.train(itera=1,  testing=True)
        elif self.config['usage_modus'] == 'fine_tuning':
            self.train(itera=5, testing=True)

        xml_file_path = self.config['folder_exp'] + self.config['file_suffix']
        xmlstr = minidom.parseString(ET.tostring(self.xml_root)).toprettyxml(indent="   ")
        with open(xml_file_path, "a") as f:
            f.write(xmlstr)

        print(xmlstr)

        return
