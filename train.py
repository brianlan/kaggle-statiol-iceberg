
# coding: utf-8

# Training
# ===

# In[1]:


import sys; sys.path.append('..')
import time
from math import ceil, floor
from os.path import join as opj, dirname

import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
import torchvision.transforms as T

from src.dataset import StatoilIcebergDataset
from src.network import Net
from src.settings import logger
from src.tensorboard_logger import Logger
from src.utils import mkdir_r
import src.torchsample.transforms as TST


# ## Define Const

# In[2]:


train_data_path = '/home/rlan/datasets/statoil-iceberg/train.json'

BASE_DIR = '/home/rlan/projects/kaggle/kaggle-statoil-iceberg'
LOG_DIR = opj(BASE_DIR, 'log')
CHECKPOINTS_PATH = opj(BASE_DIR, 'checkpoints')
MAX_EPOCH = 30
BATCH_SIZE = 256


# ## Setup Logger

# In[3]:


model_id = str(int(time.time()))
print('model_id: %s' % model_id)
tb_logger = Logger(opj(LOG_DIR, model_id))


# ## Transform

# In[9]:


transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())), T.ToPILImage(),
                       T.RandomHorizontalFlip(), T.RandomVerticalFlip(), 
                       T.ColorJitter(brightness=0.7, contrast=0.5, saturation=0.5),
                       T.ToTensor(), TST.RandomRotate(15), TST.RandomShear(15), T.ToPILImage(),
                       T.RandomResizedCrop(size=75, scale=(0.7, 1.0)), T.ToTensor(),
                       T.Lambda(lambda x: x - x.mean())]) 


# ## Dataset

# In[10]:


dataset = StatoilIcebergDataset(train_data_path, transform=transform)
loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)


# ## Network

# In[11]:


net = Net(input_channel=2).cuda() if torch.cuda.is_available() else Net(input_channel=2)
net.train()


# ## Loss and Optimizer

# In[12]:


optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = torch.nn.CrossEntropyLoss()


# ## Train

# In[13]:


niter_per_epoch = ceil(len(dataset) / BATCH_SIZE)
pbar = tqdm.tqdm(range(niter_per_epoch * MAX_EPOCH))
for epoch in pbar:
    lr_scheduler.step()
    for i_batch, sampled_batch in enumerate(loader):
        data, target = sampled_batch

        if torch.cuda.is_available():
            data, target = Variable(data).cuda(), Variable(target).cuda()
        else:
            data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        pred = net(data)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        pbar.set_description('Epoch: {:d}, Training loss: {:.4f}'.format(floor(pbar.n / niter_per_epoch), 
                                                                         loss.data[0]))
        tb_logger.scalar_summary('loss', loss.data[0], epoch * niter_per_epoch + i_batch + 1)

    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        tb_logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
        tb_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch + 1)

    if (epoch + 1) % 5 == 0:
        cp_path = opj(CHECKPOINTS_PATH, model_id, 'model_%s' % epoch)
        mkdir_r(dirname(cp_path))
        torch.save(net.state_dict(), cp_path)

