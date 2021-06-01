'''
Created on Aug 7, 2019

@author: fmoya
'''
import numpy as np
import torch
import logging


class Metrics(object):
    '''
    classdocs
    '''

    def __init__(self, config, dev, attributes):
    #def __init__(self, config, dev):
        '''
        Constructor
        '''

        logging.info('            Metrics: Constructor')
        self.config = config
        self.device = dev
        
        # Here, you need to extract the attributes from the network.pt
        #self,attr= network["att_rep"]
        self.attr = attributes
       
        #for attr_idx in range(self.attr.shape[0]):
        #    self.attr[attr_idx, 1:] = self.attr[attr_idx, 1:] / np.linalg.norm(self.attr[attr_idx, 1:])

        self.atts = torch.from_numpy(self.attr).type(dtype=torch.FloatTensor)
        #self.atts = torch.from_numpy(self.attr)
        self.atts = self.atts.type(dtype=torch.cuda.FloatTensor)
        
        if self.config['num_attributes'] == 4:
            self.center= self.atts[0:6,1:]
        elif self.config['num_attributes'] == 10:
            self.center= self.atts[0:6,1:]
            self.center= torch.cat((self.center, self.atts[7:8,1:]), 0)
        
        self.results = {'acc': 0, 'f1_weighted': 0, 'f1_mean': 0, 'predicted_classes': 0, 'precision': 0, 'recall': 0, 'acc_attrs': 0, 
                        'precision_attr': 0, 'recall_attr': 0,}

        return

    ##################################################
    ###########  Precision and Recall ################
    ##################################################
    def get_precision_recall(self, targets, predictions):
        '''
        Compute the precision and recall for all the activity classes

        @param targets: torch array with targets
        @param predictions: torch array with predictions
        @return precision: torch array with precision of each class
        @return recall: torch array with recall of each class
        '''
       
        if self.config['output'] == 'softmax':
            precision = torch.zeros((self.config['num_classes']))
            recall = torch.zeros((self.config['num_classes']))
        elif self.config['output'] == 'attribute':
            precision = torch.zeros((self.center.shape[0]))
            recall = torch.zeros((self.center.shape[0]))
       
        x = torch.ones(predictions.size())
        y = torch.zeros(predictions.size())

        x = x.to(self.device, dtype=torch.long)
        y = y.to(self.device, dtype=torch.long)
       
        if self.config['output'] == 'softmax':
            for c in range(self.config['num_classes']):
                selected_elements = torch.where(predictions == c, x, y)
                
                non_selected_elements = torch.where(predictions == c, y, x)

                target_elements = torch.where(targets == c, x, y)
                non_target_elements = torch.where(targets == c, y, x)
            
                true_positives = torch.sum(target_elements * selected_elements)
                false_positives = torch.sum(non_target_elements * selected_elements)
           
                false_negatives = torch.sum(target_elements * non_selected_elements)

                try:
                    precision[c] = true_positives.item() / float((true_positives + false_positives).item())
                    recall[c] = true_positives.item() / float((true_positives + false_negatives).item())

                except:
                    # logging.error('        Network_User:    Train:    In Class {} true_positives {} false_positives {} false_negatives {}'.format(c, true_positives.item(),
                    #                                                                                                                              false_positives.item(),
                    #                                                                                                                              false_negatives.item()))
                    continue
        elif self.config['output'] == 'attribute':
            for c in range(self.center.shape[0]):
                selected_elements = torch.where(predictions == c, x, y)
               
                non_selected_elements = torch.where(predictions == c, y, x)
               
                target_elements = torch.where(targets == c, x, y)
                non_target_elements = torch.where(targets == c, y, x)
            
                true_positives = torch.sum(target_elements * selected_elements)
                false_positives = torch.sum(non_target_elements * selected_elements)
           
                false_negatives = torch.sum(target_elements * non_selected_elements)

                try:
                    precision[c] = true_positives.item() / float((true_positives + false_positives).item())
                    recall[c] = true_positives.item() / float((true_positives + false_negatives).item())

                except:
                    # logging.error('        Network_User:    Train:    In Class {} true_positives {} false_positives {} false_negatives {}'.format(c, true_positives.item(),
                    #                                                                                                                              false_positives.item(),
                    #                                                                                                                              false_negatives.item()))
                    continue
       
        return precision, recall


    #############################################################
    ###########  Precision and Recall Attributes ################
    #############################################################

    def get_precision_recall_attrs(self, targets, predictions):
        '''
        Compute the precision and recall for all the attributes

        @param targets: torch array with targets
        @param predictions: torch array with predictions
        @return precision: torch array with precision of each class
        @return recall: torch array with recall of each class
        '''
        precision = torch.zeros((self.config['num_attributes']))
        recall = torch.zeros((self.config['num_attributes']))
       
        x = torch.ones(predictions.size()[0])
        y = torch.zeros(predictions.size()[0])
        x = x.to(self.device, dtype=torch.long)
        y = y.to(self.device, dtype=torch.long)

        for c in range(self.config['num_attributes']):
          
            selected_elements = torch.where(predictions[:, c] == 1.0, x, y)
           
            non_selected_elements = torch.where(predictions[:, c] == 1.0, y, x)

            target_elements = torch.where(targets[:, c] == 1.0, x, y)
            non_target_elements = torch.where(targets[:, c] == 1.0, y, x)

            true_positives = torch.sum(target_elements * selected_elements)
            false_positives = torch.sum(non_target_elements * selected_elements)

            false_negatives = torch.sum(target_elements * non_selected_elements)

            try:
                precision[c] = true_positives.item() / float((true_positives + false_positives).item())
                recall[c] = true_positives.item() / float((true_positives + false_negatives).item())

            except:
                # logging.error('        Network_User:    Train:    In Class {} true_positives {} false_positives {} false_negatives {}'.format(c, true_positives.item(),
                #                                                                                                                              false_positives.item(),
                #                                                                                                                              false_negatives.item()))
                continue

        return precision, recall

    ##################################################
    #################  F1 metric  ####################
    ##################################################

    def f1_metric(self, targets, preds):
        '''
        Compute the f1 metrics

        @param targets: torch array with targets
        @param predictions: torch array with predictions
        @return F1_weighted: F1 weighted
        @return F1_mean: F1 mean
        '''
        print("F1 metric")
       
        # Predictions
        if self.config['output'] == 'softmax':
            predictions = torch.argmax(preds, dim=1)
        elif self.config['output'] == 'attribute':
            # predictions = torch.argmin(preds, dim=1)
            #predictions = self.atts[torch.argmin(preds, dim=1), 0]
            #predictions = self.center[torch.argmin(preds, dim=1), 0]
            predictions=torch.argmin(preds, dim=1)
        
        if self.config['output'] == 'softmax':
            precision, recall = self.get_precision_recall(targets, predictions)
        elif self.config['output'] == 'attribute':
            precision, recall = self.get_precision_recall(targets[:, 0], predictions)
       
        if self.config['output'] == 'softmax':
            proportions = torch.zeros(self.config['num_classes'])
        elif self.config['output'] == 'attribute':
            proportions = torch.zeros(self.center.shape[0])
   
        if self.config['output'] == 'softmax':
            for c in range(self.config['num_classes']):
                proportions[c] = torch.sum(targets == c).item() / float(targets.size()[0])
        elif self.config['output'] == 'attribute':
            #for c in range(self.config['num_classes']):
            for c in range(self.center.shape[0]):
                proportions[c] = torch.sum(targets[:, 0] == c).item() / float(targets[:, 0].size()[0])
        
        logging.info('            Metric:    \nPrecision: \n{}\nRecall\n{}'.format(precision, recall))

        self.results['precision'] = precision
        self.results['recall'] = recall

        multi_pre_rec = precision * recall
        sum_pre_rec = precision + recall

        multi_pre_rec[torch.isnan(multi_pre_rec)] = 0
        sum_pre_rec[torch.isnan(sum_pre_rec)] = 0

        # F1 weighted
        weighted_f1 = proportions * (multi_pre_rec / sum_pre_rec)
        weighted_f1[np.isnan(weighted_f1)] = 0

        F1_weighted = torch.sum(weighted_f1) * 2
       
        # F1 mean
        f1 = multi_pre_rec / sum_pre_rec
        f1[torch.isnan(f1)] = 0

        if self.config['output'] == 'softmax':
            F1_mean = torch.sum(f1) * 2 / self.config['num_classes']
        elif self.config['output'] == 'attribute':
            F1_mean = torch.sum(f1) * 2 / self.center.shape[0]

        return F1_weighted.item(), F1_mean.item()

    ##################################################
    #################  Accuracy  ####################
    ##################################################

    def acc_metric(self, targets, predictions):
        '''
        Compute the Accuracy

        @param targets: torch array with targets
        @param predictions: torch array with predictions
        @return acc: Accuracy
        '''

        # Accuracy
        if self.config['output'] == 'softmax':
            predicted_classes = torch.argmax(predictions, dim=1).type(dtype=torch.cuda.FloatTensor)
            acc = torch.sum(targets == predicted_classes)
        elif self.config['output'] == 'attribute':
            # Classes from the distances to the attribute representation
            # with this torch.argmin(predictions, dim=1), one computes the argument where distance is min
            # with this self.atts[torch.argmin(predictions, dim=1), 0]
            #  one computes the class that correspond to the argument with min distance
            # self.atts.size() = [# of windows, classes and 19 attributes] = [# of windows, 20], [#, 20]
            #predicted_classes = self.atts[torch.argmin(predictions, dim=1), 0]
            predicted_classes = torch.argmin(predictions, dim=1)
            logging.info('            Metric:    Acc:    Target     class {}'.format(targets[0, 0]))
            logging.info('            Metric:    Acc:    Prediction class {}'.format(predicted_classes[0]))
            acc = torch.sum(targets[:, 0] == predicted_classes.type(dtype=torch.cuda.FloatTensor))
        elif self.config['output'] == 'identity':
            predicted_classes = torch.argmax(predictions, dim=1).type(dtype=torch.cuda.FloatTensor)
            acc = torch.sum(targets == predicted_classes)
        acc = acc.item() / float(targets.size()[0])
        
        # returning accuracy and predicted classes
        return acc, predicted_classes

    ##################################################
    ################  Acc attr  #####################
    ##################################################

    def metric_attr(self, targets, predictions):
        '''
        Compute the Accuracy per attribute or attribute vector

        @param targets: torch array with targets
        @param predictions: torch array with predictions
        @return acc_vc: Accuracy per attribute vector
        @return acc_atr: Accuracy per attribute
        '''
        
        # Accuracy per attr
        acc_attrs = np.zeros(self.config["num_attributes"])
        
        for attr_idx in range(self.config["num_attributes"]):
            acc_attrs[attr_idx] = torch.sum(torch.round(targets)[:, attr_idx] == torch.round(predictions)[:, attr_idx])
            acc_attrs[attr_idx] = acc_attrs[attr_idx] / float(targets.size()[0])
      
        logging.info('            Metric:    Acc attr: \n{}'.format(acc_attrs))
        
        precision_attr, recall_attr = self.get_precision_recall_attrs(targets, torch.round(predictions))
        logging.info('            Metric:    Precision attr: \n{}'.format(precision_attr))
        logging.info('            Metric:    Recall attr: \n{}'.format(recall_attr))
        
        #predicted_classes = self.atts[torch.argmin(predictions, dim=1), 0]
        
        return acc_attrs, precision_attr, recall_attr

    ##################################################
    ###################  metric  ######################
    ##################################################
    def efficient_distance(self, predictions):
        '''
        Compute Euclidean distance from predictions (output of sigmoid) to attribute representation

        @param predictions: torch array with predictions (output from sigmoid)
        @return distances: Euclidean Distance to each of the vectors in the attribute representation
        '''
       
        euclidean = torch.nn.PairwiseDistance()

        # Normalize the predictions of the network
        for pred_idx in range(predictions.size()[0]):
            predictions[pred_idx, :] = predictions[pred_idx,:] / torch.norm(predictions[pred_idx, :])
        
        #predictions = predictions.repeat(self.attr.shape[0], 1, 1)
        #predictions = predictions.repeat(self.atts.shape[0], 1, 1)
        
        if self.config['num_attributes'] == 4:
            predictions = predictions.repeat(6, 1, 1)
        elif self.config['num_attributes'] == 10:
            predictions = predictions.repeat(7, 1, 1)
       
        #predictions = predictions.repeat(8, 1, 1)
        predictions = predictions.permute(1, 0, 2)
        
        # compute the distance among the predictions of the network
        # and the the attribute representation
        distances = euclidean(predictions[0], self.center)
        distances = distances.view(1, -1)
        for i in range(1, predictions.shape[0]):
            dist = euclidean(predictions[i], self.center)
            distances = torch.cat((distances, dist.view(1, -1)), dim=0)
        
        #logging.info('            Metric:   distance attr: \n{}'.format(distances))
        # return the distances
        return distances

    ##################################################
    ###################  metric  ######################
    ##################################################

    def metric(self, targets, predictions):
       
        # logging.info('        Network_User:    Metrics')
        if self.config['output'] == 'attribute':
            logging.info('\n')
            logging.info('            Metric:    metric:    target example \n{}\n{}'.format(targets[0, 1:],
                                                                                            predictions[0]))
            logging.info('            Metric:    type targets vector: {}'.format(targets.type()))
            acc_attrs, precision_attr, recall_attr =self.metric_attr(targets[:, 1:], predictions)
            
            predictions = self.efficient_distance(predictions)
        print('metric')
                
        if self.config['output'] == 'attribute':
            if self.config['num_attributes'] == 4:
                for i in range(0, targets.shape[0]):
                    if targets[i,0]==6:
                        targets[i,0]=5
                    elif targets[i,0]==7:
                        targets[i,0]=3
            elif self.config['num_attributes'] == 11:
                for i in range(0, targets.shape[0]):
                    if targets[i,0]==6:
                        targets[i,0]=5
                    elif targets[i,0]==7:
                        targets[i,0]=6
       
        # Accuracy
        targets = targets.type(dtype=torch.FloatTensor)
        targets = targets.to(self.device)
        # if softmax = argmax(predictions)
        #if sigmoid = argmin(predictions) why? because predcitions are in this case distances
        #
        # with self.acc_metric, one computes the classes either from softmax os attributes
        
        acc, predicted_classes = self.acc_metric(targets, predictions)

        # F1 metrics
        f1_weighted, f1_mean = self.f1_metric(targets, predictions)

        self.results['acc'] = acc
        self.results['f1_weighted'] = f1_weighted
        self.results['f1_mean'] = f1_mean
        self.results['predicted_classes'] = predicted_classes
        self.results['acc_attrs'] = acc_attrs
        self.results['precision_attr'] = precision_attr
        self.results['recall_attr'] = recall_attr
        #return acc, f1_weighted, f1_mean, predicted_classes
        return self.results