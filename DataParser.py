import numpy as np

import scipy.sparse as sp
import numpy as np
from time import time


class Dataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, folder, data_name, batch_size):
        '''
        Constructor
        '''
        self.batch_size = int(batch_size)
        self.X, self.Y = self.read_node_label(folder + 'labels.txt')
        self.node_fea = self.read_node_features(folder + data_name + '.features')
        self.node_num = self.node_fea.shape[0]
        self.fea_dim = self.node_fea.shape[1]
        
        self.node_seq = self.read_node_sequences(folder + 'node_sequences_10_10.txt')
        
        self.node_bag_seq_dict, self.node_bag_seq_list = self.read_bag_node_sequences(folder + 'node_sequences_10_10.txt')
        self.node_list = self.read_bag_node_list(folder + 'node_sequences_10_10.txt')

        #self.num_batch = (len(self.node_bag_seq_list) / self.batch_size) + 1
        self.num_batch = (len(self.node_bag_seq_list) / self.batch_size) 


    def read_node_label(self, filename):
        fin = open(filename, 'r')
        X = []
        Y = []
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.strip().split(' ')

            if len(vec) == 2:
                X.append(int(vec[0]))
                Y.append([int(v) for v in vec[1:]])
        fin.close()
        return X, Y

    def read_node_features(self, filename):
        fea = []
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            fea.append(np.array([float(x) for x in vec[1:]]))
        fin.close()
        return np.array(fea, dtype='float32')


    def read_node_sequences(self, filename):
        seq = []
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            seq.append(np.array([int(x) for x in vec]))
        fin.close()
        return np.array(seq)

    def read_bag_node_sequences(self, filename):
        seq = {}
        seq_node_list = []
        seq_list = []
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            if int(vec[0]) not in seq:
                seq[int(vec[0])] = []
                seq_node_list.append(int(vec[0]))
            seq[int(vec[0])].append(np.array([int(x) for x in vec]))
        fin.close()
        for node in seq_node_list:
            seq_list.append(seq[node])
        return seq, seq_list

    def read_bag_node_list(self, filename):
        node_list = []
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            if int(vec[0]) not in node_list:
                node_list.append(int(vec[0]))
        fin.close()
        return np.array(node_list)

    def _preprocess(self):  # generate the masked batch list
        seq_input_list = []
        num_idx_list = []

        for i in range(int(self.num_batch)):
            seq_list_i, num_idx_list_i = self._get_train_batch_fixed(i)
            seq_input_list.append(seq_list_i)
            num_idx_list.append(num_idx_list_i)

        return [seq_input_list, num_idx_list, int(self.num_batch)]


    def _get_train_batch_fixed(self, i):
        # represent the feature of users via items rated by him/her
        seq_list = []
        num_list = []
        node_bag_seq_list = self.node_bag_seq_list
        begin = i * self.batch_size
        for idx in range(begin, min(begin + self.batch_size, len(node_bag_seq_list))):
            seq_idx = node_bag_seq_list[idx]
            seq_list.append(seq_idx)
            num_list.append([len(seq_idx)])
        return seq_list, num_list
