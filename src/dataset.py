import os
import json

import torch
import numpy as np
from torch.utils.data import Dataset


OPJ = os.path.join


class StatoilIcebergDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.transform = transform
        with open(file_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        band1 = np.array(self.data[item]['band_1']).reshape(1, 75, 75).transpose(1, 2, 0)
        band2 = np.array(self.data[item]['band_2']).reshape(1, 75, 75).transpose(1, 2, 0)
        target = self.data[item].get('is_iceberg', -1)

        if self.transform is not None:
            band1 = self.transform(band1)
            band2 = self.transform(band2)

        im = torch.cat((band1, band2))

        return im, target
