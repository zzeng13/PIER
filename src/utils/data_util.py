#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random
from transformers import BartTokenizerFast, BertTokenizerFast, AutoTokenizer
import torch
from torch.utils import data as torch_data
from src.utils.file_util import load_json_file
from config import Config
from torch.nn.utils.rnn import pad_sequence


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


class DatasetPI(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class DataHandlerCLS(object):
    # Data handler for the idiom/literal classification model

    def __init__(self):
        super(DataHandlerCLS, self).__init__()
        self.config = Config()
        self.load_data()
        self.init_generators()
        self.update_config()

    def load_data(self):
        raw_data = load_json_file(self.config.PATH_TO_CLS_TASK_DATA)
        idiom2embed = load_json_file(
            './data/magpie_idiom2embed_dictionary_google_wiki_single_meaning_cleaner.json')
        self.idioms = [k for k in idiom2embed.keys()]
        print('Number of idioms: {}'.format(len(self.idioms)))

        self.data = {'train': [d for d in raw_data['train'] if d[-1] in self.idioms],
                     'valid': [d for d in raw_data['valid'] if d[-1] in self.idioms],
                     'test': [d for d in raw_data['test'] if d[-1] in self.idioms]}

        self.tokenizer = BartTokenizerFast.from_pretrained(self.config.PRETRAINED_MODEL_NAME, do_lower_case=True,
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
        sentences, phrase_masks, labels, idioms = zip(*data)
        sentences, phrase_masks, labels, idioms = list(sentences), list(phrase_masks), list(labels), list(idioms)

        # Data pre-processing
        # masking
        # create tensors
        inputs = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True,
                                                  truncation=True)  # max_length=self.config.MAX_SEQ_LEN*2)
        phrase_masks = pad_sequence([torch.Tensor(seq) for seq in phrase_masks],
                                    batch_first=True, padding_value=0)
        phrase_masks = phrase_masks[:, :inputs['input_ids'].shape[1]]

        labels = torch.Tensor(labels)

        return {'inputs': inputs.to(self.config.DEVICE),
                'labels': labels.long().to(self.config.DEVICE),
                'phrase_masks': phrase_masks.long().to(self.config.DEVICE),
                'idioms': idioms}


class DataHandlerPI(object):

    # Data loader for paraphrase identification

    def __init__(self):
        super(DataHandlerPI, self).__init__()
        self.config = Config()
        self.load_data()
        self.init_generators()
        self.update_config()

    def load_data(self):
        self.data = load_json_file(self.config.PATH_TO_PI_TRAIN_DATA)
        self.tokenizer = BartTokenizerFast.from_pretrained(self.config.PRETRAINED_MODEL_NAME, do_lower_case=True,
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

        self.test_dataset = Dataset(self.data['valid'])
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
        sentences, labels = zip(*data)
        sentences, labels = list(sentences), list(labels)

        # Data pre-processing
        # masking
        # create tensors
        inputs = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True,
                                                  truncation=True)  # max_length=self.config.MAX_SEQ_LEN*2)


        labels = torch.Tensor(labels)

        return {'inputs': inputs.to(self.config.DEVICE),
                'labels': labels.long().to(self.config.DEVICE)}


class DataHandlerSC(object):
    # Data Loader for Sentiment Classification

    def __init__(self):
        super(DataHandlerSC, self).__init__()
        self.config = Config()
        self.load_data()
        self.init_generators()
        self.update_config()

    def load_data(self):
        self.data = load_json_file(self.config.PATH_TO_SC_TRAIN_DATA)
        self.tokenizer = BartTokenizerFast.from_pretrained(self.config.PRETRAINED_MODEL_NAME, do_lower_case=True,
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
        sentences, labels = zip(*data)
        sentences, labels = list(sentences), list(labels)

        # Data pre-processing
        # masking
        # create tensors
        inputs = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True,
                                                  truncation=True)  # max_length=self.config.MAX_SEQ_LEN*2)


        labels = torch.Tensor(labels)

        return {'inputs': inputs.to(self.config.DEVICE),
                'labels': labels.long().to(self.config.DEVICE)}


class DataHandlerCombinedDefnClsCopyMultiTempMTL(object):

    def __init__(self):
        super(DataHandlerCombinedDefnClsCopyMultiTempMTL, self).__init__()
        self.config = Config()
        self.load_data()
        self.init_generators()
        self.update_config()

    def load_data(self):
        path_to_data_files = load_json_file(self.config.PATH_TO_META_DATA)
        raw_data = load_json_file(path_to_data_files['path_to_idx_app-defn-class-copy-masked-miltitemp_data'])
        print('Loaded PIER model training data from {}'.format(path_to_data_files['path_to_idx_app-defn-class-copy-masked-miltitemp_data']))
        self.idiom2embed = load_json_file(
            './data/magpie_idiom2embed_dictionary_google_wiki_single_meaning_cleaner.json')
        self.idx2embed = load_json_file('./data/magpie_random_idx2embed_for_combined_train.json')

        self.data = {'train': [d for d in raw_data['train'] if d[-2] == 'orig' or d[-2] == 'l' and d[-3] in self.idiom2embed],
                     'valid': [d for d in raw_data['valid'] if d[-2] == 'orig' or d[-2] == 'l' and d[-3] in self.idiom2embed]}

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
                                                        batch_size=self.config.BATCH_SIZE,
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

    def collate_fn(self, data):
        source_sent, target_sent, mask, idioms, types, indices = zip(*data)
        xs, ys, mask, idioms, types = list(source_sent), list(target_sent), list(mask), list(
            idioms), list(types)
        indices = list(indices)

        # Randomly select cls prompt infilling, defn prompt infilling, or copy as the objective
        # And randomly select a template from the infilling prompt templates
        for bi in range(len(xs)):
            dice = random.randint(0, 2)
            if dice != 2:
                s_dice = random.randint(0, 3)
                xs[bi] = xs[bi][dice][s_dice]
                ys[bi] = ys[bi][dice][s_dice]
                mask[bi] = mask[bi][dice][s_dice]
            else:
                xs[bi] = xs[bi][dice]
                ys[bi] = ys[bi][dice]
                mask[bi] = mask[bi][dice]

        # Data pre-processing
        # Source sequence
        # create tensors
        inputs = self.tokenizer.batch_encode_plus(xs, return_tensors='pt', padding=True,
                                                  truncation=True, max_length=self.config.MAX_SEQ_LEN)
        # Target sequence
        ys = self.tokenizer.batch_encode_plus(ys, return_tensors='pt', padding=True, truncation=True,
                                              max_length=self.config.MAX_SEQ_LEN)['input_ids']
        # Mask sequences
        mask = pad_sequence([torch.Tensor(seq) for seq in mask],
                            batch_first=True, padding_value=0)
        mask = mask[:, :ys.shape[1]]
        types = [1 if i == 'orig' else 0 for i in types]

        defn_embed_pos = []
        for i, typ in enumerate(types):
            idiom = idioms[i]
            if typ == 0:  # 0 indicates literal sentences
                defn_embed_pos.append(torch.FloatTensor(self.idx2embed[str(indices[i])]))
            else:
                defn_embed_pos.append(torch.FloatTensor(self.idiom2embed[idiom]))
        defn_embed_pos = torch.vstack(defn_embed_pos)

        defn_embed_neg = []
        for i, typ in enumerate(types):
            idiom = idioms[i]
            if typ == 1:  # 0 indicates literal sentences
                defn_embed_neg.append(torch.FloatTensor(self.idx2embed[str(indices[i])]))
            else:
                defn_embed_neg.append(torch.FloatTensor(self.idiom2embed[idiom]))
        defn_embed_neg = torch.vstack(defn_embed_neg)

        types = torch.tensor(types)

        return {'inputs': inputs.to(self.config.DEVICE),
                'labels': ys.long().to(self.config.DEVICE),
                'idioms': idioms,
                'mask': mask.to(self.config.DEVICE),
                'defn_embed_pos': defn_embed_pos.to(self.config.DEVICE),
                'defn_embed_neg': defn_embed_neg.to(self.config.DEVICE),
                'types': types.to(self.config.DEVICE),
                'indices': indices}






















