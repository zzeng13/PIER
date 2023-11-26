#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Training/validation/inference step for PIER and models except for IE identification.
"""

import torch
from tqdm import tqdm
from src.utils.eval_util import *


def train_step(model, optimizer, data_handler, epoch, writer):
    """Train model for one epoch"""
    model.train()
    # performance recorders
    loss_epoch = AverageMeter()
    loss_cos_epoch = AverageMeter()
    acc_epoch = AverageMeter()

    # train data for a single epoch
    bbar = tqdm(enumerate(data_handler.trainset_generator), ncols=100, leave=False,
                total=data_handler.config.num_batch_train)
    mask_id = data_handler.tokenizer.mask_token_id

    for idx, data in bbar:
        # model forward pass to compute the node embeddings
        loss, loss_cos, logits = model(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # prepare for eval
        ys = data['labels'].cpu().detach().numpy()
        ys_ = torch.argmax(logits, dim=-1).cpu().detach().numpy()

        batch_size = ys.shape[0]
        loss = loss.detach().cpu().item()
        loss_epoch.update(loss, batch_size)
        loss_cos_epoch.update(loss_cos, batch_size)
        # eval results
        acc = accuracy_score(ys[ys_!=mask_id].flatten(), ys_[ys_!=mask_id].flatten())
        acc_epoch.update(acc, batch_size)
        # random sample to show
        bbar.set_description("Phase: [Train] | Train Loss: {:.5f} | Acc: {:.3f}".format(loss, acc))
        if idx % data_handler.config.SAVE_FREQ == 0 and data_handler.config.USE_TENSORBOARD:
            writer.add_scalar('train_loss', loss_epoch.avg, epoch*data_handler.config.num_batch_train+idx)
            writer.add_scalar('train_loss_cos', loss_cos_epoch.avg, epoch*data_handler.config.num_batch_train+idx)
            writer.add_scalar('train_acc', acc_epoch.avg, epoch*data_handler.config.num_batch_train+idx)
    return


def valid_step(model, data_handler):
    """Valid model for one epoch"""
    model.eval()
    torch.cuda.empty_cache()
    # performance recorders
    loss_epoch = AverageMeter()
    acc_epoch = AverageMeter()
    mask_id = data_handler.tokenizer.mask_token_id

    # valid for a single epoch
    bbar = tqdm(enumerate(data_handler.validset_generator),
                ncols=100, leave=False, total=data_handler.config.num_batch_valid)
    for idx, data in bbar:
        # model forward pass
        with torch.no_grad():
            # model forward pass to compute loss
            loss, _, logits = model(data)
        ys = data['labels'].cpu().detach().numpy()
        ys_ = torch.argmax(logits, dim=-1).cpu().detach().numpy()
        batch_size = ys.shape[0]
        loss = loss.detach().cpu().item()
        loss_epoch.update(loss, batch_size)
        # eval results
        acc = accuracy_score(ys[ys_!=mask_id].flatten(), ys_[ys_!=mask_id].flatten())
        acc_epoch.update(acc, batch_size)
        # random sample to show
        bbar.set_description("Phase: [Valid] | Valid Loss: {:.5f} | Acc: {:.3f}".format(loss, acc))

    return loss_epoch.avg, acc_epoch.avg


def test_step(model, data_handler):
    """Test model for one epoch"""
    model.eval()
    torch.cuda.empty_cache()
    # performance recorders
    loss_epoch = AverageMeter()
    acc_epoch = AverageMeter()

    # valid for a single epoch
    bbar = tqdm(enumerate(data_handler.testset_generator),
                ncols=100, leave=False, total=data_handler.config.num_batch_valid)
    for idx, data in bbar:
        # model forward pass
        with torch.no_grad():
            # model forward pass to compute loss
            loss, logits = model(data)
        ys = data['labels'].cpu().detach().numpy()
        ys_ = torch.argmax(logits, dim=-1).cpu().detach().numpy()
        batch_size = ys.shape[0]
        loss = loss.detach().cpu().item()
        loss_epoch.update(loss, batch_size)
        # eval results
        acc = accuracy_score(ys[ys_!=1].flatten(), ys_[ys_!=1].flatten())
        acc_epoch.update(acc, batch_size)
        # random sample to show
        bbar.set_description("Phase: [Test] | Valid Loss: {:.5f} | Acc: {:.3f}".format(loss, acc))

    return loss_epoch.avg, acc_epoch.avg
