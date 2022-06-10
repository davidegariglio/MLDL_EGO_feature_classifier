import torch

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, records, labels):
        'Initialization'
        self.labels = labels
        self.records = records

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.records)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        feat = self.records[index]

        # Load data and get label
        X = feat
        y = self.labels[index]

        return X, y