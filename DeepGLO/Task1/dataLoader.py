import json
import os


import torch

from torch.utils.data import Dataset

import numpy as np


class datasetCSV(Dataset):
    def __init__(self, seq_data, seq_dim):
        super(datasetCSV, self).__init__()
        self.csv_input = seq_data
        self.seq_dim = seq_dim

    def __getitem__(self, idx):
        self.input = self.csv_input[idx][0]
        # self.input = np.expand_dims(self.input, axis= 1)
        self.output = self.csv_input[idx][1]
        # self.output = np.expand_dims(self.output, axis=1)
        self.input = torch.as_tensor(np.array(self.input).astype('float'))
        self.output = torch.as_tensor(np.array(self.output).astype('float'))
        return self.input, self.output

    def __len__(self):
        return len(self.csv_input)