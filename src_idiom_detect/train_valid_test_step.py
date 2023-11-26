#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Training/validation/inference step for Seq2seq model.
"""

import torch
from tqdm import tqdm
from src_idiom_detect.utils.eval_util import *
from src_idiom_detect.utils.file_util import *


def train_step(model, optimizer, data_handler, criterion, epoch, writer):
    """Train model for one epoch"""
    model.train()
    # performance recorders
    loss_epoch = AverageMeter()
    seq_acc_epoch = AverageMeter()

    # train data for a single epoch
    bbar = tqdm(enumerate(data_handler.trainset_generator), ncols=100, leave=False, total=data_handler.config.num_batch_train)
    for idx, data in bbar:
        torch.cuda.empty_cache()
        batch_size = data['xs']['input_ids'].shape[0]

        # model forward pass to compute the node embeddings
        ys_ = model(data['xs'], data['x_lens'], data['ys'], training=True)
        if data_handler.config.DETECT_MODEL_TYPE == 'bilstm':
            data['ys'] = data['ys'][:, 1:]
        loss = criterion(ys_.reshape(-1, data_handler.config.TGT_VOCAB_SIZE), data['ys'].reshape(-1))

        # compute negative sampling loss and update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log eval metrics
        loss = loss.detach().cpu().item()
        loss_epoch.update(loss, batch_size)

        # eval results
        xs = data['xs']['input_ids'].cpu().detach().numpy()  # batch_size, max_xs_seq_len
        ys = data['ys'].cpu().detach().numpy()  # batch_size, max_ys_seq_len
        ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy()  # batch_size, max_ys_seq_len
        xs, ys, ys_ = post_process_eval(xs, ys, ys_, data_handler.config)
        seq_acc = get_seq_acc(ys, ys_)
        seq_acc_epoch.update(seq_acc, 1)
        # random sample to show
        if data_handler.config.VERBOSE and idx % data_handler.config.DISPLAY_FREQ == 0:
            src, tar, pred = rand_sample(xs, ys, ys_)
            bbar.set_description("Phase: [Train] | Batch: {}/{} | Train Loss: {:.3f} | Seq Acc: {:.3f}\n src: {}\n tgt: {}\n pred: {}\n".format(idx,
                                                                                             data_handler.config.num_batch_train,
                                                                                             loss, seq_acc,
                                                                                             src, tar, pred))

        # set display bar
        else:
            bbar.set_description("Phase: [Train] | Batch: {}/{} | Train Loss: {:.5f} | Seq Acc: {:.3f}".format(idx,
                                                                                             data_handler.config.num_batch_train,
                                                                                             loss, seq_acc))
        if idx % data_handler.config.SAVE_FREQ == 0:
            if data_handler.config.USE_TENSORBOARD:
                writer.add_scalar('train_loss', loss_epoch.avg, epoch*data_handler.config.num_batch_train+idx)
                writer.add_scalar('train_seq_acc', seq_acc, epoch*data_handler.config.num_batch_train+idx)

    return loss_epoch.avg, seq_acc_epoch.avg


def valid_step(model, data_handler, criterion):
    """Valid model for one epoch"""
    model.eval()
    torch.cuda.empty_cache()
    # performance recorders
    loss_epoch = AverageMeter()
    seq_acc_epoch = AverageMeter()
    preds = []
    labels = []
    # valid for a single epoch
    bbar = tqdm(enumerate(data_handler.validset_generator), ncols=100, leave=False, total=data_handler.config.num_batch_valid)
    for idx, data in bbar:
        torch.cuda.empty_cache()
        batch_size = data['xs']['input_ids'].shape[0]

        # model forward pass
        with torch.no_grad():
            # model forward pass to compute loss
            ys_, _ = model(data['xs'], data['x_lens'], data['ys'], training=False)
            if data_handler.config.DETECT_MODEL_TYPE == 'bilstm':
                data['ys'] = data['ys'][:, 1:]
            loss = criterion(ys_.reshape(-1, data_handler.config.TGT_VOCAB_SIZE), data['ys'].reshape(-1))

        # log eval metrics
        loss = loss.detach().cpu().item()
        loss_epoch.update(loss, batch_size)

        # eval results
        xs = data['xs']['input_ids'].cpu().detach().numpy()  # batch_size, max_xs_seq_len
        ys = data['ys'].cpu().detach().numpy()  # batch_size, max_ys_seq_len
        ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy()  # batch_size, max_ys_seq_len
        xs, ys, ys_ = post_process_eval(xs, ys, ys_, data_handler.config)
        seq_acc = get_seq_acc(ys, ys_)
        seq_acc_epoch.update(seq_acc, batch_size)
        preds += list(ys_)
        labels += list(ys)

        # random sample to show
        if data_handler.config.VERBOSE:
            src, tar, pred = rand_sample(xs, ys, ys_)
            bbar.set_description("Phase: [Valid] | Batch: {}/{} | Valid Loss: {:.3f} | Seq Acc: {:.3f}\n src: {}\n tgt: {}\n pred: {}\n".format(idx,
                                                                                             data_handler.config.num_batch_valid,
                                                                                             loss,
                                                                                             seq_acc,
                                                                                             src, tar, pred))

        # set display bar
        else:
            bbar.set_description("Phase: [Valid] | Batch: {}/{} | Valid Loss: {:.3f} | Seq Acc: {:.3f}".format(idx,
                                                                                          data_handler.config.num_batch_valid,
                                                                                          loss,
                                                                                          seq_acc))
    return loss_epoch.avg, get_seq_acc(labels, preds)


def eval_step(model, data_handler):
    """Evaluate model for one epoch"""
    model.eval()
    torch.cuda.empty_cache()
    # performance recorders
    seq_acc_epoch = AverageMeter()
    cls_tars, cls_preds = [], []
    y_preds = []
    attn_scores_epoch = []
    target_seqs = []
    predict_seqs = []
    # valid for a single epoch
    bbar = tqdm(enumerate(data_handler.testset_generator), ncols=100, leave=True)
    for idx, data in bbar:
        torch.cuda.empty_cache()

        # model forward pass
        with torch.no_grad():
            # model forward pass to compute loss
            ys_, attn_scores = model(data['xs'], data['x_lens'], data['ys'])
            if data_handler.config.MODEL_TYPE == 'bilstm':
                data['ys'] = data['ys'][:, 1:]

        # eval results
        batch_size = data['xs']['input_ids'].shape[0]
        xs = data['xs']['input_ids'].cpu().detach().numpy()  # batch_size, max_xs_seq_len
        ys = data['ys'].cpu().detach().numpy()  # batch_size, max_ys_seq_len

        attn_scores_epoch.append(list(attn_scores.cpu().detach().numpy()))
        ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy()  # batch_size, max_ys_seq_len
        xs, ys, ys_ = post_process_eval(xs, ys, ys_, data_handler.config)
        target_seqs.append(ys)
        predict_seqs.append(ys_)
        seq_acc, correctness = get_seq_acc_test(ys, ys_)
        cur_cls_tars, cur_cls_preds = get_cls_acc_test(ys, ys_)
        cls_tars += cur_cls_tars
        cls_preds += cur_cls_preds


        seq_acc_epoch.update(seq_acc, batch_size)

        # format data
        for i in range(len(ys_)):
            ys_[i] = [s.item() for s in ys_[i]]
        for i, c in enumerate(correctness):
            y_preds.append([data['id'][i], c])


        # random sample to show
        bbar.set_description("Phase: [Test] | Batch: {}/{} | Seq Acc: {:.3f}".format(idx,
                                                                                      data_handler.config.num_batch_test,
                                                                                    seq_acc))
    performance_tok = get_seq_performance(target_seqs, predict_seqs)
    performance_tok['seq_acc'] = seq_acc_epoch.avg
    performance_cls = get_cls_performance(cls_tars, cls_preds)
    write_json_file('./res/semeval5b_random_seq_performance.json', {'seq_correctness': y_preds})

    return performance_tok, performance_cls, attn_scores_epoch


def test_step(model, data_handler, criterion):
    """test model for one epoch"""
    model.eval()
    torch.cuda.empty_cache()
    # performance recorders
    loss_epoch = AverageMeter()
    seq_acc_epoch = AverageMeter()
    preds = []
    labels = []

    # valid for a single epoch
    bbar = tqdm(enumerate(data_handler.testset_generator), ncols=100, leave=False)
    for idx, data in bbar:
        torch.cuda.empty_cache()
        batch_size = data['xs']['input_ids'].shape[0]

        # model forward pass
        with torch.no_grad():
            # model forward pass to compute loss
            ys_, _ = model(data['xs'], data['x_lens'], data['ys'], training=False)
            if data_handler.config.DETECT_MODEL_TYPE == 'bilstm':
                data['ys'] = data['ys'][:, 1:]
            loss = criterion(ys_.reshape(-1, data_handler.config.TGT_VOCAB_SIZE), data['ys'].reshape(-1))

        # log eval metrics
        loss = loss.detach().cpu().item()
        loss_epoch.update(loss, batch_size)

        # eval results
        xs = data['xs']['input_ids'].cpu().detach().numpy()  # batch_size, max_xs_seq_len
        ys = data['ys'].cpu().detach().numpy()  # batch_size, max_ys_seq_len
        ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy()  # batch_size, max_ys_seq_len
        xs, ys, ys_ = post_process_eval(xs, ys, ys_, data_handler.config)
        seq_acc = get_seq_acc(ys, ys_)
        seq_acc_epoch.update(seq_acc, batch_size)
        preds += list(ys_)
        labels += list(ys)

        # random sample to show
        if data_handler.config.VERBOSE:
            src, tar, pred = rand_sample(xs, ys, ys_)
            bbar.set_description("Phase: [Valid] | Batch: {}/{} | Valid Loss: {:.3f} | Seq Acc: {:.3f}\n src: {}\n tgt: {}\n pred: {}\n".format(idx,
                                                                                             data_handler.config.num_batch_valid,
                                                                                             loss,
                                                                                             seq_acc,
                                                                                             src, tar, pred))

        # set display bar
        else:
            bbar.set_description("Phase: [Test] | Batch: {}/{} | Valid Loss: {:.3f} | Seq Acc: {:.3f}".format(idx,
                                                                                          data_handler.config.num_batch_valid,
                                                                                          loss,
                                                                                          seq_acc))
    return loss_epoch.avg, get_seq_acc(labels, preds)

