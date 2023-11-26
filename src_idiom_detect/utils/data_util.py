#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Data packing and processing.
"""

import math
import torch
from torch.utils import data as torch_data
from torch.nn.utils.rnn import pad_sequence
from src_idiom_detect.utils.file_util import *
from config import Config
from transformers import BartTokenizerFast, BertTokenizerFast


# Data handler for training and validation data
class Dataset(torch_data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, xs):
        super(Dataset, self).__init__()
        self.xs = xs
        self.num_total_seqs = len(self.xs)

    def __len__(self):
        return self.num_total_seqs

    def __getitem__(self, index):
        return self.xs[index]


class DataHandler(object):

    def __init__(self):
        super(DataHandler, self).__init__()
        self.config = Config()
        self.load_data()
        self.init_generators()
        self.update_config()

    def load_data(self):
        if 'bert' in self.config.MODEL_TYPE:
            raw_data = load_json_file(self.config.PATH_TO_DET_TASK_DATA_BERT)

        else:
            raw_data = load_json_file(self.config.PATH_TO_DET_TASK_DATA)
        idiom2embed = load_json_file('./data/magpie_idiom2embed_dictionary_google_wiki_single_meaning_cleaner.json')
        self.idioms = [k for k in idiom2embed.keys()]
        self.data = {'train': [d for d in raw_data['train'] if d[-1] in self.idioms],
                    'valid': [d for d in raw_data['valid'] if d[-1] in self.idioms],
                     'test': [d for d in raw_data['test'] if d[-1] in self.idioms]}
        if 'bert' in self.config.MODEL_TYPE:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.config.PRETRAINED_BERT_NAME,
                                                           do_lower_case=True,
                                                           add_prefix_space=True)
        else:
            self.tokenizer = BartTokenizerFast.from_pretrained(self.config.PRETRAINED_MODEL_NAME,
                                                           do_lower_case=True,
                                                           add_prefix_space=True)

    def init_generators(self):
        self.train_dataset = Dataset(self.data['train'])
        self.trainset_generator = torch_data.DataLoader(self.train_dataset,
                                                        batch_size=self.config.BATCH_SIZE,
                                                        collate_fn=self.collate_fn,
                                                        shuffle=True,
                                                        num_workers=self.config.NUM_WORKER,
                                                        drop_last=True)
        # data loader for validset
        self.valid_dataset = Dataset(self.data['valid'])
        self.validset_generator = torch_data.DataLoader(self.valid_dataset,
                                                        batch_size=self.config.BATCH_SIZE * 2,
                                                        collate_fn=self.collate_fn,
                                                        shuffle=False,
                                                        num_workers=self.config.NUM_WORKER,
                                                        drop_last=False)

        self.test_dataset = Dataset(self.data['test'])
        self.testset_generator = torch_data.DataLoader(self.test_dataset,
                                                        batch_size=self.config.BATCH_SIZE * 2,
                                                        collate_fn=self.collate_fn,
                                                        shuffle=False,
                                                        num_workers=self.config.NUM_WORKER,
                                                        drop_last=False)

    def update_config(self):
        def get_batch_size(dataset_size):
            if dataset_size % self.config.BATCH_SIZE == 0:
                return dataset_size // self.config.BATCH_SIZE
            else:
                return dataset_size // self.config.BATCH_SIZE + 1

        # training parameters
        self.config.train_size = len(self.train_dataset)
        self.config.valid_size = len(self.valid_dataset)
        print('Training dataset size: {}'.format(self.config.train_size))
        print('Validation dataset size: {}'.format(self.config.valid_size))
        self.config.num_batch_train = get_batch_size(self.config.train_size)
        self.config.num_batch_valid = get_batch_size(self.config.valid_size)
        self.config.test_size = len(self.test_dataset)
        print('Testing dataset size: {}'.format(self.config.test_size))
        self.config.num_batch_test = get_batch_size(self.config.test_size)

    def collate_fn(self, data):
        sentences, phrase_masks, labels, _ = zip(*data)
        sentences, phrase_masks, labels = list(sentences), list(phrase_masks), list(labels)

        # Data pre-processing
        # masking
        # create tensors
        inputs = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True,
                                                  truncation=True, max_length=self.config.MAX_SEQ_LEN)
        xs_lens = torch.sum(inputs['attention_mask'], dim=-1)
        phrase_masks = pad_sequence([torch.Tensor(seq) for seq in phrase_masks],
                                    batch_first=True, padding_value=0)
        phrase_masks = phrase_masks[:, :inputs['input_ids'].shape[1]]

        labels = torch.Tensor(labels)

        return {'xs': inputs.to(self.config.DEVICE),
                'labels': labels.long().to(self.config.DEVICE),
                'x_lens': xs_lens.long().to(self.config.DEVICE),
                'ys': phrase_masks.long().to(self.config.DEVICE)}
