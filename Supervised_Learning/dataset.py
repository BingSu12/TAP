
import torch
import torch.utils.data as data
import numpy as np
#from PIL import Image

import os
#from DataSet import transforms
from collections import defaultdict

import scipy.io as scio


class TrainData(data.Dataset):
    def __init__(self, root=None, data_mat=None):
        datapath = os.path.join(root,data_mat)
        Dismatdata = scio.loadmat(datapath)
        sequence_data = Dismatdata['trainseqfull']
        sequence_data = sequence_data.copy(order='C')
        sequence_data = sequence_data.astype(np.float32)
        sequence_label = Dismatdata['trainlabelseqfull']
        sequence_label = sequence_label.copy(order='C')
        #sequence_label = sequence_label.astype(np.float32)
        sequence_length = Dismatdata['trainlengthseqfull']
        sequence_length = sequence_length.copy(order='C')
        #sequence_length = sequence_length.astype(np.float32)

        sequences = []
        for i in range(sequence_data.shape[0]):
            sequences.append(sequence_data[i,:,:])
        labels = []
        intlabels = []
        for i in range(sequence_label.shape[0]):
            labels.append(sequence_label[i,:])
            intlabels.append(int(sequence_label[i,sequence_label.shape[1]-1]))
        lengths = []
        for i in range(sequence_length.shape[0]):
            lengths.append(int(sequence_length[i,sequence_length.shape[1]-1]))
        #print(sequences)
        print(sequences[0].shape)
        print(len(intlabels))
        print(len(lengths))

        classes = list(set(intlabels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(intlabels):
            Index[label].append(i)

        # Initialization Done
        self.root = root
        self.sequences = sequences
        self.labels = labels
        self.classes = classes
        self.intlabels = intlabels
        self.lengths = lengths
        #self.transform = transform
        self.Index = Index
        #self.loader = loader

    def __getitem__(self, index):
        sequence, intlabel, length = self.sequences[index], self.intlabels[index], self.lengths[index]
        return sequence, intlabel, length

    def __len__(self):
        return len(self.sequences)


class TestData(data.Dataset):
    def __init__(self, root=None, data_mat=None):
        datapath = os.path.join(root,data_mat)
        Dismatdata = scio.loadmat(datapath)
        sequence_data = Dismatdata['testseqfull']
        sequence_data = sequence_data.copy(order='C')
        sequence_data = sequence_data.astype(np.float32)
        sequence_label = Dismatdata['testlabelseqfull']
        sequence_label = sequence_label.copy(order='C')
        sequence_length = Dismatdata['testlengthseqfull']
        sequence_length = sequence_length.copy(order='C')
        #sequence_length = sequence_length.astype(np.float32)

        sequences = []
        for i in range(sequence_data.shape[0]):
            sequences.append(sequence_data[i,:,:])
        labels = []
        intlabels = []
        for i in range(sequence_label.shape[0]):
            labels.append(sequence_label[i,:])
            intlabels.append(int(sequence_label[i,sequence_label.shape[1]-1]))
        lengths = []
        for i in range(sequence_length.shape[0]):
            lengths.append(int(sequence_length[i,sequence_length.shape[1]-1]))
        #print(sequences)
        print(sequences[0].shape)
        print(len(intlabels))
        print(len(lengths))

        classes = list(set(intlabels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(intlabels):
            Index[label].append(i)

        # Initialization Done
        self.root = root
        self.sequences = sequences
        self.labels = labels
        self.classes = classes
        self.intlabels = intlabels
        self.lengths = lengths
        #self.transform = transform
        self.Index = Index
        #self.loader = loader

    def __getitem__(self, index):
        sequence, intlabel, length = self.sequences[index], self.intlabels[index], self.lengths[index]
        return sequence, intlabel, length

    def __len__(self):
        return len(self.sequences)


class SequenceDataset:
    def __init__(self, root=None):
        train_mat = "trainseqfull.mat"
        test_mat = "testseqfull.mat"
        if root is None:
            root = './'
        self.traindata = TrainData(root, data_mat=train_mat)
        self.testdata = TestData(root, data_mat=test_mat)

        # if train_flag:
        #     self.seqdata = TrainData(root, data_mat=train_mat)
        # else:
        #     self.seqdata = TestData(root, data_mat=test_mat)


def testsequence():
    print(SequenceDataset.__name__)
    data = SequenceDataset()
    print(len(data.testdata))
    print(len(data.traindata))
    print(data.traindata[1])


if __name__ == "__main__":
    testsequence()


