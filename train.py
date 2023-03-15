# Train One Experiment
# Leon Cai
# MASI Lab
# July 21, 2022

# Set Up

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np
import math
import os

from utils import default_context
from data import DT1Dataset, unload
from modules import RecurrentModel, CosineLoss

# Helper Functions

def epoch_losses(data_loader, fod_cnn, t1_cnn, fod_rnn, t1_rnn, cnn_criterion, mid_criterion, fc_criterion, rnn_criterion, opt, label, stage):

    train = 'train' in label.lower()

    epoch_loss         = 0
    epoch_fod_dot_loss = 0
    epoch_fod_cum_loss = 0
    epoch_t1_dot_loss  = 0
    epoch_t1_cum_loss  = 0
    epoch_fod_loss     = 0
    epoch_fc_loss      = 0
    epoch_mid_loss     = 0

    epoch_iters = len(data_loader)
    
    for _, dt1_item in tqdm(enumerate(data_loader), total=epoch_iters, desc=label, leave=False):

        # ten, step, trid, trii, mask, tdi = unload(*dt1_item)
        # ten, fod, brain, step, trid, trii, mask = unload(*dt1_item)
        ten_1mm, ten_2mm, fod, brain, step, trid, trii, mask = unload(*dt1_item)

        if train:
            fod_cnn.train()
            t1_cnn.train()
            fod_rnn.train()
            t1_rnn.train()
            opt.zero_grad()
        else:
            fod_cnn.eval()
            t1_cnn.eval()
            fod_rnn.eval()
            t1_rnn.eval()
            
        if stage == 0:
            loss_fxn = lambda fod_dot_loss, fod_cum_loss, t1_dot_loss, t1_cum_loss, fod_loss, fc_loss, mid_loss : fod_dot_loss
        elif stage == 1:
            loss_fxn = lambda fod_dot_loss, fod_cum_loss, t1_dot_loss, t1_cum_loss, fod_loss, fc_loss, mid_loss : fc_loss + t1_dot_loss # + mid_loss

        with (default_context() if (train and stage == 0) else torch.no_grad()):
            fod_pred = fod_cnn(fod.to(device))
            fod_step_pred, _, _, _, fod_fc = fod_rnn(fod_pred, trid.to(device), trii)
            fod_dot_loss, fod_cum_loss = rnn_criterion(fod_step_pred, step.to(device), mask.to(device))
        with (default_context() if (train and stage == 1) else torch.no_grad()):
            t1_pred = t1_cnn(ten_1mm.to(device), ten_2mm.to(device))
            t1_step_pred, _, _, _, t1_fc = t1_rnn(t1_pred, trid.to(device), trii)
            t1_dot_loss, t1_cum_loss = rnn_criterion(t1_step_pred, step.to(device), mask.to(device))
            fod_loss = torch.Tensor([0]) # cnn_criterion(t1_pred, fod_pred, brain.to(device))
            fc_loss = fc_criterion(t1_fc, fod_fc)
            # fc_loss, _ = fc_criterion(t1_step_pred, fod_step_pred, mask.to(device)) # *** trying "end loss" to impose contrastive at output instead of in the middle
            mid_loss = torch.Tensor([0]) # mid_criterion(t1_fc, fod_fc)
        loss = loss_fxn(fod_dot_loss, fod_cum_loss, t1_dot_loss, t1_cum_loss, fod_loss, fc_loss, mid_loss)

        if train:
            loss.backward()
            opt.step()
        
        epoch_loss         += loss.item()
        epoch_fod_dot_loss += fod_dot_loss.item()
        epoch_fod_cum_loss += fod_cum_loss.item()
        epoch_t1_dot_loss  += t1_dot_loss.item()
        epoch_t1_cum_loss  += t1_cum_loss.item()
        epoch_fod_loss     += fod_loss.item()
        epoch_fc_loss      += fc_loss.item()
        epoch_mid_loss     += mid_loss.item()

    epoch_loss         /= epoch_iters
    epoch_fod_dot_loss /= epoch_iters
    epoch_fod_cum_loss /= epoch_iters
    epoch_t1_dot_loss  /= epoch_iters
    epoch_t1_cum_loss  /= epoch_iters
    epoch_fod_loss     /= epoch_iters
    epoch_fc_loss      /= epoch_iters
    epoch_mid_loss     /= epoch_iters

    return epoch_loss, epoch_fod_dot_loss, epoch_fod_cum_loss, epoch_t1_dot_loss, epoch_t1_cum_loss, epoch_fod_loss, epoch_fc_loss, epoch_mid_loss

# Inputs

out_dir = 'tensorboard/pilot_hcp_100_bestfodrnn_t1k7_1mm2mm'
weights_file = os.path.join(out_dir, 'fod_rnn.pt')

learn_dirs_file = 'learn_dirs.txt'
test_dirs_file = 'test_dirs.txt'

num_streamlines = 1000000
batch_size = 1000
num_batches = np.ceil(num_streamlines / batch_size).astype(int)

# Prepare data

with open(learn_dirs_file, 'r') as learn_dirs_fobj:
    learn_dirs = learn_dirs_fobj.read().splitlines()
train_dirs = learn_dirs[:80]
val_dirs = learn_dirs[80:]
with open(test_dirs_file, 'r') as test_dirs_fobj:
    test_dirs = test_dirs_fobj.read().splitlines()

train_dataset = DT1Dataset(train_dirs, num_batches)
val_dataset   = DT1Dataset(val_dirs,   num_batches)
test_dataset  = DT1Dataset(test_dirs,  num_batches)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=1, num_workers=1, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=1, num_workers=1, shuffle=True)

img_shape = (96, 114, 96)

# Train

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

fod_rnn = RecurrentModel(45, fc_width=512, fc_depth=4, rnn_width=512, rnn_depth=2).to(device)
rnn_criterion = CosineLoss()
opt = optim.Adam(list(fod_rnn.parameters()), lr=1e-3)

total_max_epochs = 1000
stage_num_epochs_no_change = 100

best_loss = math.inf
best_epoch = '-'

writer = SummaryWriter(out_dir)
dot2ang = lambda dot : 180/np.pi*np.arccos(1-dot)
epoch_bar = tqdm(range(total_max_epochs), leave=True)

for epoch in epoch_bar:

    epoch_bar.set_description('Stage: {} | Best Epoch: {} | Current Epoch'.format(stage, best_epoch))

    train_loss, train_fod_dot_loss, train_fod_cum_loss, train_t1_dot_loss, train_t1_cum_loss, train_fod_loss, train_fc_loss, train_mid_loss = epoch_losses(train_loader, fod_cnn, t1_cnn, fod_rnn, t1_rnn, cnn_criterion, mid_criterion, fc_criterion, rnn_criterion, opts[stage], 'Train', stage)
    val_loss, val_fod_dot_loss, val_fod_cum_loss, val_t1_dot_loss, val_t1_cum_loss, val_fod_loss, val_fc_loss, val_mid_loss = epoch_losses(val_loader, fod_cnn, t1_cnn, fod_rnn, t1_rnn, cnn_criterion, mid_criterion, fc_criterion, rnn_criterion, opts[stage], 'Validation', stage)

    train_fod_ang_loss = dot2ang(train_fod_dot_loss)
    val_fod_ang_loss = dot2ang(val_fod_dot_loss)
    train_t1_ang_loss = dot2ang(train_t1_dot_loss)
    val_t1_ang_loss = dot2ang(val_t1_dot_loss)

    writer.add_scalars('Total Loss',     {'Train': train_loss,         'Validation': val_loss},         epoch)
    writer.add_scalars('FOD Dot Loss',   {'Train': train_fod_dot_loss, 'Validation': val_fod_dot_loss}, epoch)
    writer.add_scalars('FOD Angle Loss', {'Train': train_fod_ang_loss, 'Validation': val_fod_ang_loss}, epoch)
    writer.add_scalars('FOD Cum Loss',   {'Train': train_fod_cum_loss, 'Validation': val_fod_cum_loss}, epoch)
    writer.add_scalars('T1 Dot Loss',    {'Train': train_t1_dot_loss,  'Validation': val_t1_dot_loss},  epoch)
    writer.add_scalars('T1 Angle Loss',  {'Train': train_t1_ang_loss,  'Validation': val_t1_ang_loss},  epoch)
    writer.add_scalars('T1 Cum Loss',    {'Train': train_t1_cum_loss,  'Validation': val_t1_cum_loss},  epoch)
    writer.add_scalars('FOD Loss',       {'Train': train_fod_loss,     'Validation': val_fod_loss},     epoch)
    writer.add_scalars('FC Loss',        {'Train': train_fc_loss,      'Validation': val_fc_loss},      epoch)
    writer.add_scalars('Mid Loss',       {'Train': train_mid_loss,     'Validation': val_mid_loss},     epoch)

    if val_loss < best_loss:
        torch.save(fod_cnn.state_dict(), fod_cnn_file)
        torch.save(fod_rnn.state_dict(), fod_rnn_file)
        torch.save(t1_cnn.state_dict(), t1_cnn_file)
        torch.save(t1_rnn.state_dict(), t1_rnn_file)
        best_loss = val_loss
        best_epoch = epoch
        stage_last_epoch = np.min((epoch + stage_num_epochs_no_change - 1, stage_max_epochs - 1, total_max_epochs - 1)) # always computed in first epoch of new stage since best_loss = inf; -1 to account for 0 indexing

    if epoch == stage_last_epoch:
        print('\nStage: {} | Best Epoch: {} | Last Epoch: {}'.format(stage, best_epoch, epoch))
        stage += 1
        if stage == 1:
            fod_cnn.load_state_dict(torch.load(fod_cnn_file, map_location=device))
            fod_rnn_weights = torch.load(fod_rnn_file, map_location=device)
            fod_rnn.load_state_dict(fod_rnn_weights)
            for weights_name in list(fod_rnn_weights.keys()):
                if 'fc' in weights_name:
                    del fod_rnn_weights[weights_name]
            t1_rnn.load_state_dict(fod_rnn_weights, strict=False) # load weights from fod_rnn!
            for param in list(t1_rnn.rnn.parameters()) + list(t1_rnn.azi.parameters()) + list(t1_rnn.ele.parameters()):
                param.requires_grad = False
        if stage > 1:
            break
        stage_max_epochs = epoch + stage_max_epochs + 1 # +1 to account for -1 above for zero indexing
        best_loss = math.inf
        best_epoch = '-'

    # schs[stage].step()
