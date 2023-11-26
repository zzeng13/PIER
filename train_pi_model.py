#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Training script for paraphrase identification model.
"""
import os
import random
import numpy as np
from datetime import datetime
from tqdm import trange
from tensorboardX import SummaryWriter
from src.utils.data_util import DataHandlerPI
from src.train_valid_test_step_tasks import *
from config import Config
from torch.multiprocessing import set_start_method
from src.classifiers.paraphrase_identifier import ParaphraseIdentifier


__author__ = "Ziheng Zeng"
__email__ = "zzeng13@illinois.edu"
__status__ = "Prototype"

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_model(model, epoch, optimizer, type, save_path):
    save_path = save_path.format(type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path += 'projection_layer.mdl'
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    # save the best model seen so far
    torch.save(state, save_path)


def train():
    """
    The training script for the Relation2Vec model.
    """
    # Initialize a data loader
    # ---------------------------------------------------------------------------------
    data_handler = DataHandlerPI()

    # Manage and initialize model
    # ---------------------------------------------------------------------------------
    # Initialize model
    epoch_start = 0  #
    model = ParaphraseIdentifier(data_handler.config)
    model.to(data_handler.config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=data_handler.config.LEARNING_RATE)
    # freeze adapters
    # for param in model.adapter.parameters():
    #     param.requires_grad = False
    print('------------------------------------------------')
    print('[TRAINABLE PARAMETERS]: ')

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print('------------------------------------------------')
    #
    # Add Tensorboard writer
    writer = None
    if Config.USE_TENSORBOARD:
        writer = SummaryWriter(log_dir='./runs_pi_new/{}_{}'.format(Config.PI_NAME,
                                                                   datetime.today().strftime('%Y-%m-%d')))
    # Book-keeping info
    best_valid_acc = float('-inf')


    # Train model
    # ---------------------------------------------------------------------------------
    ebar = trange(epoch_start, Config.NUM_EPOCHS, desc='EPOCH', ncols=130, leave=True)
    set_seed(Config.SEED)

    for epoch in ebar:
        # Training
        train_step(model, optimizer, data_handler, epoch, writer)

        # Validation
        if epoch % Config.VALID_FREQ == 0:
            valid_loss, valid_acc = valid_step(model, data_handler)
            print('\n Validation Acc: {}'.format(valid_acc))
            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                # save the best model seen so far
                model.save_model('best')
            if Config.USE_TENSORBOARD:
                writer.add_scalar('valid_loss', valid_loss, epoch)
                writer.add_scalar('valid_acc', valid_acc, epoch)

        # save the latest model
        model.save_model('latest')
        # model.save_model('latest')
    return


if __name__ == '__main__':
    train()


