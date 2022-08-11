# IMPORTS
# ______________________________________________________________________________________________________________________
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import os


# LIBRARY
# ______________________________________________________________________________________________________________________

class CustomDataset(Dataset):

    def __init__(self, dataframe, tensors, n, name):
        """
        :param dataframe: the firm-year sample dataset that should be transformed into a PyTorch dataset
        :param tensors: the numpy array containing the sorted document representations (according to the doc_ids)
        :param n: fixed-length time window
        :param name: the model used 'bow' for word2vec or 'transformers'
        """

        # initialize the variables
        self.tensors = tensors
        self.n = n
        self.name = name

        # retain the columns in the dataframe that contain doc_ids
        # these are the columns 'year_...' (but not 'holdout_year')
        doc_id_columns = []
        cols = dataframe.columns
        for col in cols:
            if ('year' in col) & ('hol' not in col):
                doc_id_columns.append(col)

        # assign these columns and the labels
        self.data = dataframe[doc_id_columns]
        self.labels = torch.FloatTensor(dataframe['label'])

    def __len__(self):
        """
        :return: the number of sample in the dataset
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        :param idx: the index of the element from the dataset to retrieve
        :return: the sample as a tuple with the sample itself as a tensor, the label and the index
        """

        # store the doc_ids of the element to retrieve in a list
        doc_ids = list(self.data.loc[idx])

        # loop over the doc ids and store the corresponding encoding in a list
        sample = []
        for doc_id in doc_ids:

            # missing document - use all zero vector as encoding
            if doc_id == 'missing':
                if self.name == 'transformers':
                    encoding = np.zeros(768)
                else:
                    encoding = np.zeros(100)

            # non-missing documents: retrieve encoding
            else:
                encoding = np.array(self.tensors[int(doc_id)])

            # add encoding to sample
            sample.append(torch.FloatTensor(encoding))

        # format encoding and cast to tensor
        sample = torch.cat(sample)
        if self.name == 'transformers':
            sample = sample.reshape([768, self.n])
        else:
            sample = sample.reshape([100, self.n])

        # retrieve the label
        label = self.labels[idx]

        return (sample, label, idx)

# ______________________________________________________________________________________________________________________
