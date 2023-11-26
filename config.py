#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Experiment Configuration.
"""
from os.path import join, abspath, dirname
import torch


class Config:
    ROOT = abspath(dirname(__file__))

    # Settings - (regularly changed)
    # ==============================================================================================================
    MODE = 'train'  # 'train' or 'test'
    ADAPTER_NAME = 'fusion'  # "fusion" to run PIER models; "non-compositional" to run GIEA models.
    DATA_NAME = 'magpie'
    SPLIT = 'random'
    MODEL_TYPE = 'bart-adapters'
    TRAIN_DATA_NAMES = sorted(["magpie-rd"])
    # TRAIN_DATA_NAMES = sorted(["mrpc"])
    MODEL_NAME = '{}_{}_{}_{}'.format(MODEL_TYPE, ADAPTER_NAME, DATA_NAME, SPLIT)
    # point to the training data of the PIER model.
    PATH_TO_META_DATA = './meta_data/meta_data_magpie_{}_mtl.json'.format(SPLIT)
    USE_MTL = True  # only useful during training; Set 'True' to use the similarity forcing objective.
    USE_ADAPTER = True  # set true for all training using adapter, set false only for fine-tuning bart
    USE_COMBINED = True  # only useful during training of the application cls and det model; Set True to use PIER as backbone (BartAdapterCombined model)

    DET_TYPE = 'random'
    CLS_TYPE = 'random'
    # Model names for the downstream tasks.
    CLF_NAME = '{}_{}-PIER'.format('Literal-Idiomatic', MODEL_NAME)
    DET_NAME = '{}_{}-PIER'.format('IdiomDetectSeq2Seq', MODEL_NAME)
    PI_NAME = '{}_{}-PIER'.format('ParaphraseIdentification', MODEL_NAME)
    SC_NAME = '{}_{}-PIER'.format('SentimentClassification', MODEL_NAME)
    USE_BART_AS_COMPOSITIONAL = True  # always set true for PIER model; setting True to use BART as the compositional encoder

    # Training configs
    USE_GPU = True
    CONTINUE_TRAIN = True  # Set to False when training the PIER model (unless a checkpoint exist); Set to True when using combined modules for NLP/idiom tasks
    USE_TENSORBOARD = True
    VERBOSE = False  # display sampled prediction results
    NUM_WORKER = 0
    SEED = 42
    MASK_SPAN_RATIO = 0.8
    MAX_SEQ_LEN = 128
    LOSS_MASK_SCALE = 0.3

    # Checkpoint management
    # checkpoint name for PIER model.
    PATH_TO_CHECKPOINT = join(ROOT, f'checkpoints/{MODEL_NAME}-PIER' + '/{}/')
    # Checkpoint names for span detection and disambiguation task
    PATH_TO_CHECKPOINT_DET = join(ROOT, f'checkpoints/SpanDetection/{DET_NAME}-{DET_TYPE}' + '/{}/')
    PATH_TO_CHECKPOINT_CLS = join(ROOT, f'checkpoints/Literal-Idiomatic-CLS/{MODEL_NAME}' + '/{}/')
    PATH_TO_SAVE_PRETRAINED = join(ROOT, './bart-base-ft/{}/')
    # Set to load the latest or the best checkpoint. Default to be 'best'
    LOAD_CHECKPOINT_TYPE = 'best'  # 'latest' or 'best

    # ++++++++++++++++++++++++++++++++++++++++++ PARAMETERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Train Parameters
    # ==============================================================================================================
    NUM_EPOCHS = 30
    BATCH_SIZE = 16
    VALID_FREQ = 1  # number of epochs to run validation
    SAVE_FREQ = 10  # number of steps to save train performance
    DISPLAY_FREQ = 10  # number of steps to display train performance (only matters if VERBOSE==TRUE)
    LEARNING_RATE = 1e-5

    # Inference Parameters
    # ==============================================================================================================
    PATH_TO_SAVE_RESULTS = join(ROOT, 'res/{}_test_results.json'.format(MODEL_NAME))

    # Model Parameters
    # ==============================================================================================================
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu")
    # BART
    # Store the pre-trained models in the following directories
    PRETRAINED_MODEL_NAME = 'facebook/bart-base'
    PRETRAINED_BERT_NAME = './bert-base-uncased'
    PRETRAINED_HIDDEN_SIZE = 768

    # Adapters
    ADAPTER_REDUCTION_FACTOR = 16
    ADAPTER_ARCHITECTURE_NAME = 'pfeiffer'

    # Classifiers
    # Literal-Idiomatic
    CLS_LI_NUM_CLASSES = 2
    LAYER_TYPE = 'last'  # 'last' || 'all' || 'decoder-only'
    # paraphrase identification
    CLS_PI_NUM_CLASSES =2

    # Settings for the Span Detection model (Idiom Detect)
    DETECT_MODEL_TYPE = 'bilstm-direct'
    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    TGT_VOCAB_SIZE = 5
    PRETRAINED_EMBED_DIM = 768
    # Seq2seq
    EMBEDDING_DIM = 256
    ENCODER_HIDDEN_DIM = 256
    DECODER_HIDDEN_DIM = 256
    # Bi-LSTM
    BILSTM_ENCODER_NUM_LAYERS = 1
    BILSTM_DECODER_NUM_LAYERS = 1
    BILSTM_ENCODER_DROP_RATE = 0.3
    BILSTM_DECODER_DROP_RATE = 0.3
    BILSTM_LINEAR_DROP_RATE = 0.2
    BILSTM_TEACHER_FORCING_RATIO = 0.7

    # Data for disambiguation task: Literal idiomatic classification
    PATH_TO_CLS_TASK_DATA = './data/idiomatic_literal_calssification_processed_data_magpie_{}.json'.format(CLS_TYPE)
    PATH_TO_CHECKPOINT_CLF = join(ROOT, f'checkpoints/{CLF_NAME}' + f'/{ADAPTER_NAME}-{SPLIT}-{CLS_TYPE}'+'/{}/')
    PATH_TO_CLS_TASK_DATA_BERT = './data/idiomatic_literal_classification_bert_{}.json'.format(CLS_TYPE)

    # Data for Span detection task: a.k.a. IE identification task.
    PATH_TO_DET_TASK_DATA = './data/MAGPIE_{}_noContext_literal_idiomatic_detection_for_BART_adapters.json'.format(
        DET_TYPE)

    # Data for paraphrase identification task
    PATH_TO_PI_TRAIN_DATA = './data/pi_mrpc_paws_data.json'
    PATH_TO_CHECKPOINT_PI = join(ROOT, f'checkpoints/{PI_NAME}' + f'/{ADAPTER_NAME}-{SPLIT}' + '/{}/')

    # Data for sentiment classification task
    PATH_TO_SC_TRAIN_DATA = './data/sst2_sentiment_classification_data.json'
    PATH_TO_CHECKPOINT_SC = join(ROOT, f'checkpoints/{SC_NAME}' + f'/{ADAPTER_NAME}-{SPLIT}' + '/{}/')

    # Mode based parameters
    if MODE == 'test' or MODE == 'inference':
        CONTINUE_TRAIN = True
        USE_TENSORBOARD = False
        BATCH_SIZE = 4


