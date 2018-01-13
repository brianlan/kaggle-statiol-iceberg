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

        # import torchvision.transforms as T
        # import src.torchsample.transforms as TST
        # tmp_transform = T.Compose(
        #     [T.ToTensor(),
        #      T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
        #      T.ToPILImage(),
        #      T.RandomHorizontalFlip(),
        #      T.RandomVerticalFlip(),
        #      # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        #      T.ToTensor(),
        #      TST.RandomRotate(5),
        #      # TST.RandomShear(15),
        #      T.ToPILImage(),
        #      T.RandomResizedCrop(size=75, scale=(0.7, 1.0)),
        #      T.ToTensor(),
        #      T.Lambda(lambda x: x - 0.5),
        #      ])

        if self.transform is not None:
            band1 = self.transform(band1)
            band2 = self.transform(band2)

        # import torchvision.transforms.functional as F
        # save_dir = '/home/rlan/projects/kaggle/kaggle-statoil-iceberg/output_images/augmented'
        # F.to_pil_image(band1 + 0.5).save(OPJ(save_dir, str(target), self.data[item]['id'] + '_band1.png'))
        # F.to_pil_image(band2 + 0.5).save(OPJ(save_dir, str(target), self.data[item]['id'] + '_band2.png'))

        im = torch.cat((band1, band2))

        return im, target
