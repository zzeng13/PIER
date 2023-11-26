import os
from adapter_variants.src.transformers.models.bert import BertForMaskedLM, BertConfig
from adapter_variants.src.transformers.adapters.composition import Fuse, Flow
from adapter_variants.src.transformers.adapters.configuration import AdapterConfig
from torch import nn
import torch


class BertAdapter(nn.Module):
    def __init__(self, config):
        super(BertAdapter, self).__init__()
        self.config = config
        self.vocab_size = BertConfig().vocab_size
        # Base BART Model
        self.model = BertForMaskedLM.from_pretrained(config.PRETRAINED_BERT_NAME,  output_hidden_states=True)
        # Adapters
        adapter_config = AdapterConfig.load(config=config.ADAPTER_ARCHITECTURE_NAME,
                                            reduction_factor=config.ADAPTER_REDUCTION_FACTOR)
        # add and activate adapters
        self.attach_adapter(adapter_config)
        self.model.to(config.DEVICE)
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def mean_pooling(self, token_embeddings, attention_mask):
        # token_embeddings: [batch_size, max_seq_len, hidden_dim]
        # token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        if token_embeddings.shape[1] != attention_mask.shape[1]:
            print('Shape mismatch occurred! Embed Shape: {} || Mask Shape: {}'.format(token_embeddings.shape, attention_mask.shape))
            attention_mask = torch.ones(token_embeddings.shape[0], token_embeddings.shape[1]).to(self.config.DEVICE)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def forward(self, data, writer=None):
        if self.config.MODE != 'inference':
            if self.config.USE_MTL:
                outputs = self.model(**data['inputs'], labels=data['labels'])
                l_recon = outputs.loss
                # compute the cosine embedding loss
                last_hidden_state = outputs.hidden_states[-1]
                target = torch.ones(last_hidden_state.shape[0],).to(self.config.DEVICE)
                defn_orig = data['defn_embed']
                l_cos = self.cosine_loss(self.mean_pooling(last_hidden_state, data['mask']),
                                         defn_orig,
                                         target)

                loss = l_recon + l_cos
                return loss, l_cos, outputs.logits

            else:
                outputs = self.model(**data['inputs'], labels=data['labels'])
                return outputs.loss, 0, outputs.logits

        else:
            outputs = self.model(**data['inputs'], labels=data['labels'])
            return outputs

    def attach_adapter(self, adapter_config):
        if self.config.ADAPTER_NAME == 'fusion':
            print('=> Initializing Adapters with Fusion Module...')
            # load trained adapters
            # 1. load compositional adapters
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapters
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'non-compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # Add a fusion layer for all loaded adapters
            print("==> Adding fusion module!")
            adapter_setup = Fuse("compositional", "non-compositional")
            print(adapter_setup)
            if self.config.MODE == 'train':
                if self.config.CONTINUE_TRAIN:
                    print("==> Loading Adapter Fusion from {}".format(save_path))
                    save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                    self.model.load_adapter_fusion(save_path)
                    # self.model.load_head(save_path)
                else:
                    self.model.add_adapter_fusion(adapter_setup)
                    # self.model.add_tagging_head("fusion", num_labels=self.vocab_size)
                self.model.set_active_adapters(adapter_setup)
                self.model.train_adapter_fusion(adapter_setup)
            else:
                save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                print("==> Loading Adapter Fusion from {}".format(save_path))
                self.model.load_adapter_fusion(save_path)
                adapter_setup = Fuse("compositional", "non-compositional")
                self.model.set_active_adapters(adapter_setup)
                # print("==> Loading Classification Head from {}".format(save_path))
                # self.model.load_head(save_path)
                self.model.train_adapter_fusion(adapter_setup)
                self.model.eval()

        elif self.config.ADAPTER_NAME == 'flow':
            print('=> Initializing Adapters with Flow Module...')
            # load trained adapters
            # 1. load compositional adapters
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapters
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'non-compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # Add a fusion layer for all loaded adapters
            print("==> Adding Flow module!")
            adapter_setup = Flow("compositional", "non-compositional")
            print(adapter_setup)
            if self.config.MODE == 'train':
                if self.config.CONTINUE_TRAIN:
                    print("==> Loading Adapter Flow from {}".format(save_path))
                    save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                    self.model.load_adapter_flow(save_path)
                    # self.model.load_head(save_path)
                else:
                    self.model.add_adapter_flow(adapter_setup)
                    # self.model.add_tagging_head("fusion", num_labels=self.vocab_size)
                self.model.set_active_adapters(adapter_setup)
                self.model.train_adapter_flow(adapter_setup)
            else:
                save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                print("==> Loading Adapter Flow from {}".format(save_path))
                self.model.load_adapter_flow(save_path)
                adapter_setup = Flow("compositional", "non-compositional")
                self.model.set_active_adapters(adapter_setup)
                # print("==> Loading Classification Head from {}".format(save_path))
                # self.model.load_head(save_path)
                self.model.train_adapter_flow(adapter_setup)
                self.model.eval()

        else:
            if self.config.MODE == 'inference':
                adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, self.config.ADAPTER_NAME,
                                                    self.config.DATA_NAME, self.config.SPLIT)
                save_path = os.path.join(self.config.ROOT,
                                         f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
                print("==> Loading Adapter from {}".format(save_path))
                adapter_name = self.model.load_adapter(save_path)
                # print(f"===> Set {adapter_name} to be active!")
                self.model.train_adapter(adapter_name)
                # self.model.set_active_adapters(adapter_name)
                self.model.eval()
            else:
                save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                if self.config.CONTINUE_TRAIN and os.path.exists(save_path):
                    print("==> Loading trained Adapter from {}".format(save_path))
                    adapter_name = self.model.load_adapter(save_path)
                    print("==> Trained adapter loaded from {}".format(save_path))
                else:
                    print("==> Initialize new Adapter and train from scratch!")
                    adapter_name = self.config.ADAPTER_NAME
                    # self.model.add_tagging_head(adapter_name, num_labels=self.vocab_size)
                    self.model.add_adapter(adapter_name, config=adapter_config)
                if self.config.MODE == "train":
                    self.model.train_adapter(adapter_name)
                else:
                    self.model.set_active_adapters(adapter_name)
                    self.model.eval()

    def save_model(self, save_type):
        save_path = self.config.PATH_TO_CHECKPOINT.format(save_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.config.ADAPTER_NAME == 'fusion':
            self.model.save_adapter_fusion(save_path, "compositional,non-compositional")
        elif 'flow' in self.config.ADAPTER_NAME:
            self.model.save_adapter_flow(save_path, "compositional,non-compositional")
        else:
            self.model.save_adapter(save_path, self.config.ADAPTER_NAME)


# ======================================================================================================================
class BertAdapterOld(nn.Module):
    def __init__(self, config):
        super(BertAdapterOld, self).__init__()
        self.config = config
        self.vocab_size = BertConfig().vocab_size
        # Base BART Model
        self.model = BertForMaskedLM.from_pretrained(config.PRETRAINED_MODEL_NAME)
        # Adapters
        adapter_config = AdapterConfig.load(config=config.ADAPTER_ARCHITECTURE_NAME,
                                            reduction_factor=config.ADAPTER_REDUCTION_FACTOR)
        # add and activate adapters
        self.attach_adapter(adapter_config)
        self.model.to(config.DEVICE)

    def forward(self, data):
        if self.config.MODE == 'train':
            outputs = self.model(**data['inputs'], labels=data['labels'])
            return outputs.loss, outputs.logits

        else:
            outputs = self.model(**data['inputs'], labels=data['labels'])
            return outputs

    def attach_adapter(self, adapter_config):
        if self.config.ADAPTER_NAME == 'fusion':
            print('=> Initializing Adapters with Fusion Module...')
            # load trained adapters
            # 1. load compositional adapters
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapters
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'non-compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # Add a fusion layer for all loaded adapters
            print("==> Adding fusion module!")
            adapter_setup = Fuse("compositional", "non-compositional")
            print(adapter_setup)
            if self.config.MODE == 'train':
                if self.config.CONTINUE_TRAIN:
                    print("==> Loading Adapter Fusion from {}".format(save_path))
                    save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                    self.model.load_adapter_fusion(save_path)
                    # self.model.load_head(save_path)
                else:
                    self.model.add_adapter_fusion(adapter_setup)
                    # self.model.add_tagging_head("fusion", num_labels=self.vocab_size)
                self.model.set_active_adapters(adapter_setup)
                self.model.train_adapter_fusion(adapter_setup)
            else:
                save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                print("==> Loading Adapter Fusion from {}".format(save_path))
                self.model.load_adapter_fusion(save_path)
                adapter_setup = Fuse("compositional", "non-compositional")
                self.model.set_active_adapters(adapter_setup)
                # print("==> Loading Classification Head from {}".format(save_path))
                # self.model.load_head(save_path)
                self.model.train_adapter_fusion(adapter_setup)
                self.model.eval()

        elif self.config.ADAPTER_NAME == 'flow':
            print('=> Initializing Adapters with Flow Module...')
            # load trained adapters
            # 1. load compositional adapters
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapters
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'non-compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # Add a fusion layer for all loaded adapters
            print("==> Adding Flow module!")
            adapter_setup = Flow("compositional", "non-compositional")
            print(adapter_setup)
            if self.config.MODE == 'train':
                if self.config.CONTINUE_TRAIN:
                    print("==> Loading Adapter Flow from {}".format(save_path))
                    save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                    self.model.load_adapter_flow(save_path)
                    # self.model.load_head(save_path)
                else:
                    self.model.add_adapter_flow(adapter_setup)
                    # self.model.add_tagging_head("fusion", num_labels=self.vocab_size)
                self.model.set_active_adapters(adapter_setup)
                self.model.train_adapter_flow(adapter_setup)
            else:
                save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                print("==> Loading Adapter Flow from {}".format(save_path))
                self.model.load_adapter_flow(save_path)
                adapter_setup = Flow("compositional", "non-compositional")
                self.model.set_active_adapters(adapter_setup)
                # print("==> Loading Classification Head from {}".format(save_path))
                # self.model.load_head(save_path)
                self.model.train_adapter_flow(adapter_setup)
                self.model.eval()

        else:
            if self.config.MODE == 'inference':
                adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, self.config.ADAPTER_NAME,
                                                    self.config.DATA_NAME, self.config.SPLIT)
                save_path = os.path.join(self.config.ROOT,
                                         f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
                print("==> Loading Adapter from {}".format(save_path))
                adapter_name = self.model.load_adapter(save_path)
                # print(f"===> Set {adapter_name} to be active!")
                self.model.train_adapter(adapter_name)
                # self.model.set_active_adapters(adapter_name)
                self.model.eval()
            else:
                save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                if self.config.CONTINUE_TRAIN and os.path.exists(save_path):
                    print("==> Loading trained Adapter from {}".format(save_path))
                    adapter_name = self.model.load_adapter(save_path)
                else:
                    print("==> Initialize new Adapter and train from scratch!")
                    adapter_name = self.config.ADAPTER_NAME
                    # self.model.add_tagging_head(adapter_name, num_labels=self.vocab_size)
                    self.model.add_adapter(adapter_name, config=adapter_config)
                if self.config.MODE == "train":
                    self.model.train_adapter(adapter_name)
                else:
                    self.model.set_active_adapters(adapter_name)
                    self.model.eval()

    def save_model(self, save_type):
        save_path = self.config.PATH_TO_CHECKPOINT.format(save_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.config.ADAPTER_NAME == 'fusion':
            self.model.save_adapter_fusion(save_path, "compositional,non-compositional")
        elif self.config.ADAPTER_NAME == 'flow':
            self.model.save_adapter_flow(save_path, "compositional,non-compositional")
        else:
            self.model.save_adapter(save_path, self.config.ADAPTER_NAME)




