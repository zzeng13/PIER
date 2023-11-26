#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Training script for PIER model.
"""
from datetime import datetime
import random
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter
from src.train_valid_test_step import *
from config import Config
from torch.multiprocessing import set_start_method
from src.model.bart_adapters import BartAdapterCombinedBase, BartAdapterCombined
from src.utils.data_util import DataHandlerCombinedDefnClsCopyMultiTempMTL


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


def train():
    """
    The training script for the Fusion or Flow model.
    """
    # Initialize a data loader
    # ---------------------------------------------------------------------------------
    data_handler = DataHandlerCombinedDefnClsCopyMultiTempMTL()

    # Manage and initialize model
    # ---------------------------------------------------------------------------------
    # Initialize model
    epoch_start = 0  # 0
    model = BartAdapterCombined(data_handler.config)
    optimizer = torch.optim.Adam(model.parameters(), lr=data_handler.config.LEARNING_RATE)
    # Add Tensorboard writer
    writer = None
    if Config.USE_TENSORBOARD:
        writer = SummaryWriter(log_dir='./runs-pier/{}-PIER-{}'.format(Config.MODEL_NAME, datetime.today().strftime('%Y-%m-%d')))
    best_valid_loss = float('inf')

    print('------------------------------------------------')
    print('[TRAINABLE ADAPTERS]: ')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print('------------------------------------------------')

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
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                # save the best model seen so far
                model.save_model("best")
            if Config.USE_TENSORBOARD:
                writer.add_scalar('valid_loss', valid_loss, epoch)
                writer.add_scalar('valid_acc', valid_acc, epoch)

        # save the latest model
        model.save_model("latest")
    return


if __name__ == '__main__':
    train()


