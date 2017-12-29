import os
import json

import numpy as np
from torch.utils.data import Dataset


OPJ = os.path.join


class StatoilIcebergDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.transform = transform
        with open(file_path, 'r') as f:
            self.train_data = json.load(f)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        im = np.array([self.train_data[item]['band_1'], self.train_data[item]['band_1']])
        im = im.reshape(2, 75, 75).transpose(1, 2, 0)
        target = self.train_data[item]['is_iceberg']

        if self.transform is not None:
            im = self.transform(im)

        return im, target
