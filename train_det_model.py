#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Training script for IE identification model.
"""
from datetime import datetime
from tqdm import trange
from tensorboardX import SummaryWriter
from src_idiom_detect.utils.data_util import *
from src_idiom_detect.utils.model_util import *
from src_idiom_detect.train_valid_test_step import *
from config import Config
# handle multi-processing for data loader


from src_idiom_detect.model.bilstm import Seq2SeqBiLSTMLite as Seq2SeqMdl


__author__ = "Ziheng Zeng"
__email__ = "zzeng13@illinois.edu"
__status__ = "Prototype"



def train():
    """
    The training script for the Relation2Vec model.
    """
    # Initialize a data loader
    # ---------------------------------------------------------------------------------
    data_handler = DataHandler()

    # Manage and initialize model
    # ---------------------------------------------------------------------------------
    # Initialize model
    save_path = Config.PATH_TO_CHECKPOINT_DET
    seq2seq, optimizer, epoch_start = load_init_det_model(Seq2SeqMdl, data_handler.config)
    seq2seq.to(Config.DEVICE)
    # Freeze the pre-trained embedding layer
    for param in seq2seq.src_embedding_layer.parameters():
        param.requires_grad = False
    print('------------------------------------------------')
    print('[TRAINABLE ADAPTERS]: ')

    for name, param in seq2seq.named_parameters():
        if param.requires_grad:
            print(name)
    print('------------------------------------------------')
    criterion = torch.nn.NLLLoss(ignore_index=data_handler.config.PAD_IDX)
    # Add Tensorboard writer
    writer = None
    if Config.USE_TENSORBOARD:
        writer = SummaryWriter(log_dir='./runs_det_new/{}_noContext_{}_c'.format(Config.MODEL_NAME, datetime.today().strftime('%Y-%m-%d')))
    # Book-keeping info
    best_valid_acc = float('-inf')
    print(save_path.format('best'))
    # Train model
    # ---------------------------------------------------------------------------------
    ebar = trange(epoch_start, Config.NUM_EPOCHS, desc='EPOCH', ncols=130, leave=True)
    for epoch in ebar:
        # Training
        _, _ = train_step(seq2seq, optimizer, data_handler, criterion, epoch, writer)

        # Validation
        if epoch % Config.VALID_FREQ == 0:
            valid_loss, valid_seq_acc = valid_step(seq2seq, data_handler, criterion)
            if best_valid_acc < valid_seq_acc:
                best_valid_acc = valid_seq_acc
                # save the best model seen so far
                save_model(save_path.format('best'), seq2seq, optimizer, epoch)
            if Config.USE_TENSORBOARD:
                writer.add_scalar('valid_loss', valid_loss, epoch)
                writer.add_scalar('valid_seq_acc', valid_seq_acc, epoch)
                print(valid_seq_acc)
        # save the latest model
        save_model(save_path.format('latest'), seq2seq, optimizer, epoch)

    return


if __name__ == '__main__':
    train()


