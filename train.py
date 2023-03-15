# Train One Experiment
# Leon Cai
# MASI Lab
# July 21, 2022

# Set Up

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np
import math
import sys
import os

from data import STrUDeLDataset, unload
from modules import RecurrentModel, CosineLoss

# Inputs

data_dirs_file  = sys.argv[1]       # data.txt
num_streamlines = int(sys.argv[2])  # 1000000
batch_size      = int(sys.argv[3])  # 1000
tboard_dir      = sys.argv[4]       # tensorboard
num_epochs      = int(sys.argv[5])  # 500

# Outputs

weights_file = 'weights_tutorial.pt'

# Prepare data

num_batches = np.ceil(num_streamlines / batch_size).astype(int)
with open(data_dirs_file, 'r') as data_dirs_fobj:
    data_dirs = data_dirs_fobj.read().splitlines()
data_set = STrUDeLDataset(data_dirs, num_batches)
data_loader = DataLoader(data_set, batch_size=1, num_workers=0, shuffle=True)

if not os.path.exists(tboard_dir):
    os.mkdir(tboard_dir)

# Train

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
rnn    = RecurrentModel(45, fc_width=512, fc_depth=4, rnn_width=512, rnn_depth=2).to(device)
loss   = CosineLoss()
opt    = optim.Adam(list(rnn.parameters()), lr=1e-3)

best_loss = math.inf
best_epoch = '-'

writer = SummaryWriter(tboard_dir)
epoch_bar = tqdm(range(num_epochs), leave=True)

for epoch in epoch_bar:

    epoch_bar.set_description('Best Epoch: {} | Current Epoch'.format(best_epoch))

    epoch_loss  = 0
    epoch_iters = len(data_loader)
    for _, item in tqdm(enumerate(data_loader), total=epoch_iters, desc='Training', leave=False):
        fod, step, trid, trii, mask = unload(*item)
        rnn.train()
        opt.zero_grad()
        step_pred, _, _, _ = rnn(fod.to(device), trid.to(device), trii)
        batch_loss = loss(step_pred, step.to(device), mask.to(device))
        batch_loss.backward()
        opt.step()
        epoch_loss += batch_loss.item()
    epoch_loss /= epoch_iters

    writer.add_scalar('Loss', epoch_loss, epoch)

    if epoch_loss < best_loss:
        torch.save(rnn.state_dict(), weights_file)
        best_loss = epoch_loss
        best_epoch = epoch
