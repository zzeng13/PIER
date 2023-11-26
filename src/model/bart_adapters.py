import os
from adapter_variants.src.transformers.models.bart import BartForConditionalGeneration, BartConfig, BartModelWithHeads
from adapter_variants.src.transformers.adapters.composition import Fuse, Flow, Stack
from adapter_variants.src.transformers.adapters.configuration import AdapterConfig, PfeifferInvConfig
import torch.nn.functional as F

from torch import nn
import torch


class BartAdapter(nn.Module):
    # Generic Bart Adapter for GIEA model
    def __init__(self, config):
        super(BartAdapter, self).__init__()
        self.config = config
        self.vocab_size = BartConfig().vocab_size
        # Base BART Model
        print("Load base model from {}".format(config.PRETRAINED_MODEL_NAME))
        self.model = BartForConditionalGeneration.from_pretrained(config.PRETRAINED_MODEL_NAME)
        # Adapters
        # add and activate adapters
        if config.USE_ADAPTER:
            adapter_config = AdapterConfig.load(config=config.ADAPTER_ARCHITECTURE_NAME,
                                                reduction_factor=config.ADAPTER_REDUCTION_FACTOR)
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
                target = torch.ones(outputs.last_hidden_state.shape[0],).to(self.config.DEVICE)
                defn_orig = data['defn_embed']
                l_cos = self.cosine_loss(self.mean_pooling(outputs.last_hidden_state, data['mask']),
                                         defn_orig,
                                         target)

                loss = l_recon + l_cos
                return loss, l_cos, outputs.logits
            else:
                outputs = self.model(**data['inputs'], labels=data['labels'])
                return outputs.loss, 0, outputs.logits

        else:
            outputs = self.model.generate(**data['inputs'], num_beams=5, max_length=150)
            return outputs

    def attach_adapter(self, adapter_config):
        if self.config.ADAPTER_NAME == 'fusion':
            print('=> Initializing Adapters with Fusion Module...')
            # load trained adapters
            # 1. load compositional adapters
            if self.config.USE_BART_AS_COMPOSITIONAL:
                print('==> Using BART output as compositional embedding!')
                self.model.add_adapter("compositional", config='skip')
            else:

                adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                    self.config.DATA_NAME, self.config.SPLIT)
                save_path = os.path.join(self.config.ROOT,
                                         f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapters
            adapter_name = '{}_{}_{}_{}-GIEA'.format(self.config.MODEL_TYPE, 'non-compositional',
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

        elif 'flow' in self.config.ADAPTER_NAME:
            print('=> Initializing Adapters with Flow Module...')
            # load trained adapters
            # 1. load compositional adapters
            if self.config.USE_BART_AS_COMPOSITIONAL:
                print('==> Using BART output as compositional embedding!')
                self.model.add_adapter("compositional", config='skip')
            else:

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
                if os.path.exists(save_path):
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


class BartAdapterCLS(nn.Module):
    # BART model with adapter for classification tasks, e.g., sentiment classification
    def __init__(self, config):
        super(BartAdapterCLS, self).__init__()
        self.config = config
        self.vocab_size = BartConfig().vocab_size
        # Base BART Model
        print("Load base model from {}".format(config.PRETRAINED_MODEL_NAME))
        self.model = BartModelWithHeads.from_pretrained(config.PRETRAINED_MODEL_NAME)
        # Adapters
        adapter_config = AdapterConfig.load(config=config.ADAPTER_ARCHITECTURE_NAME,
                                            reduction_factor=config.ADAPTER_REDUCTION_FACTOR)

        self.attach_adapter(adapter_config)
        self.model.to(config.DEVICE)
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def mean_pooling(self, token_embeddings, attention_mask):
        # token_embeddings: [batch_size, max_seq_len, hidden_dim]
        # token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, data, writer=None):
        outputs = self.model(**data['inputs'], labels=data['labels'])
        return outputs.loss, 0, outputs.logits

    def attach_adapter(self, adapter_config):

        if self.config.MODE == 'inference':
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, self.config.ADAPTER_NAME,
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            adapter_name = self.model.load_adapter(save_path, with_head=False)
            # print(f"===> Set {adapter_name} to be active!")
            self.model.train_adapter(adapter_name)
            # self.model.set_active_adapters(adapter_name)
            self.model.eval()
        else:
            save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
            if os.path.exists(save_path):
                print("==> Loading trained Adapter from {}".format(save_path))
                adapter_name = self.model.load_adapter(save_path, with_head=True)
                print("==> Trained adapter loaded from {}".format(save_path))
            else:
                print("==> Initialize new Adapter and train from scratch!")
                adapter_name = self.config.ADAPTER_NAME
                # self.model.add_tagging_head(adapter_name, num_labels=self.vocab_size)
                self.model.add_adapter(adapter_name, config=adapter_config)
                self.model.add_classification_head(adapter_name, num_labels=self.config.CLS_LI_NUM_CLASSES)
            if self.config.MODE == "train":
                self.model.train_adapter(adapter_name)
            else:
                self.model.set_active_adapters(adapter_name)
                self.model.eval()

    def save_model(self, save_type):
        save_path = self.config.PATH_TO_CHECKPOINT.format(save_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.save_adapter(save_path, self.config.ADAPTER_NAME)


class BartAdapterCombined(nn.Module):
    """
    PIER model with both copy reconstruction loss and similarity forcing. 
    This the final version of PIER model (PIER+)
    """
    def __init__(self, config):
        super(BartAdapterCombined, self).__init__()
        self.config = config
        self.vocab_size = BartConfig().vocab_size
        # Base BART Model
        print("Load base model from {}".format(config.PRETRAINED_MODEL_NAME))
        self.model = BartForConditionalGeneration.from_pretrained(config.PRETRAINED_MODEL_NAME)
        # Adapters
        adapter_config = AdapterConfig.load(config=config.ADAPTER_ARCHITECTURE_NAME,
                                            reduction_factor=config.ADAPTER_REDUCTION_FACTOR)
        self.attach_adapter(adapter_config)
        self.model.to(config.DEVICE)
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.nll_loss = nn.NLLLoss()

    def mean_pooling(self, token_embeddings, attention_mask):
        # token_embeddings: [batch_size, max_seq_len, hidden_dim]
        if token_embeddings.shape[1] != attention_mask.shape[1]:
            print('Shape mismatch occurred! Embed Shape: {} || Mask Shape: {}'.format(token_embeddings.shape, attention_mask.shape))
            attention_mask = torch.ones(token_embeddings.shape[0], token_embeddings.shape[1]).to(self.config.DEVICE)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def forward(self, data, writer=None):
        if self.config.MODE != 'inference':
            outputs = self.model(**data['inputs'], labels=data['labels'])
            # Reconstruction loss for infilling objectives
            l_recon = outputs.loss
            # Compute the cosine embedding loss for similarity forcing
            defn_pos = data['defn_embed_pos']
            defn_neg = data['defn_embed_neg']
            target_pos = torch.ones(outputs.last_hidden_state.shape[0], ).to(self.config.DEVICE)
            target_neg = torch.ones(outputs.last_hidden_state.shape[0], ).to(self.config.DEVICE) * -1
            ie_embeddings = self.mean_pooling(outputs.last_hidden_state, data['mask'])
            l_cos_pos = self.cosine_loss(ie_embeddings,
                                     defn_pos,
                                     target_pos)
            l_cos_neg = self.cosine_loss(ie_embeddings,
                                      defn_neg,
                                      target_neg)

            l_cos = l_cos_pos + l_cos_neg
            loss = l_recon + l_cos
            return loss, l_cos, outputs.logits

        else:
            outputs = self.model.generate(**data,eos_token_id=1, num_beams=5, max_length=350)
            return outputs

    def attach_adapter(self, adapter_config):
        if self.config.ADAPTER_NAME == 'fusion-cls':
            print('=> Initializing Adapters with Fusion Module with CLS adapter...')
            # load trained adapters
            # 1. load compositional adapters
            if self.config.USE_BART_AS_COMPOSITIONAL:
                print("\n[COMPOSITIONAL MODULE]: ")
                print('==> Using BART output as compositional embedding!')
                self.model.add_adapter("compositional", config='skip')
            else:
                adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                    self.config.DATA_NAME, self.config.SPLIT)
                save_path = os.path.join(self.config.ROOT,
                                         f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapters
            print("\n[NON-COMPOSITIONAL MODULE]: ")
            adapter_name = '{}_{}_{}_{}-GIEA'.format(self.config.MODEL_TYPE, 'non-compositional',
                                                             self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            print("==> Non-compositional adapter loaded!")
            # 3. load disambiguation adapter
            print("\n[DISAMBIGUATION MODULE]: ")
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'disambiguate',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            print("==> Non-compositional adapter loaded!")

            # Add a fusion layer for all loaded adapters
            print("\n[COMBINED MODULE]: ")
            print("==> Adding fusion module!")
            adapter_setup = Fuse("compositional", "non-compositional", "disambiguate")
            print(adapter_setup)
            if self.config.MODE == 'train':
                if self.config.CONTINUE_TRAIN:
                    save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                    print("==> Loading Adapter Fusion from {}".format(save_path))
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
                adapter_setup = Fuse("compositional", "non-compositional", "disambiguate")
                self.model.set_active_adapters(adapter_setup)
                # print("==> Loading Classification Head from {}".format(save_path))
                # self.model.load_head(save_path)
                self.model.train_adapter_fusion(adapter_setup)
                self.model.eval()

        elif self.config.ADAPTER_NAME == 'fusion':
            print('=> Initializing Adapters with Fusion Module...')
            # load trained adapters
            # 1. load compositional adapters
            if self.config.USE_BART_AS_COMPOSITIONAL:
                print("\n[COMPOSITIONAL MODULE]: ")
                print('==> Using BART output as compositional embedding!')
                self.model.add_adapter("compositional", config='skip')
            else:
                adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                    self.config.DATA_NAME, self.config.SPLIT)
                save_path = os.path.join(self.config.ROOT,
                                         f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapters (Load a trained GIEA model)
            print("\n[NON-COMPOSITIONAL MODULE]: ")
            adapter_name = '{}_{}_{}_{}-GIEA'.format(self.config.MODEL_TYPE, 'non-compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            print("==> Non-compositional adapter loaded!")
            # Add a fusion layer for all loaded adapters
            print("\n[COMBINED MODULE]: ")
            print("==> Adding fusion module!")
            adapter_setup = Fuse("compositional", "non-compositional")
            print(adapter_setup)
            if self.config.MODE == 'train':
                if self.config.CONTINUE_TRAIN:
                    save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                    print("==> Loading Adapter Fusion from {}".format(save_path))
                    self.model.load_adapter_fusion(save_path)
                    print("==>  Adapter Fusion Loaded!")
                else:
                    self.model.add_adapter_fusion(adapter_setup)
                self.model.set_active_adapters(adapter_setup)
                self.model.train_adapter_fusion(adapter_setup)
            else:
                save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                print("==> Loading Adapter Fusion from {}".format(save_path))
                self.model.load_adapter_fusion(save_path)
                adapter_setup = Fuse("compositional", "non-compositional")
                self.model.set_active_adapters(adapter_setup)
                self.model.train_adapter_fusion(adapter_setup)
                self.model.eval()

        elif 'flow' in self.config.ADAPTER_NAME:
            print('=> Initializing Adapters with <Flow> Module...')
            # load trained adapters
            # 1. load compositional adapters
            if self.config.USE_BART_AS_COMPOSITIONAL:
                print("\n[COMPOSITIONAL MODULE]: ")
                print('==> Using BART output as compositional embedding!')
                self.model.add_adapter("compositional", config='skip')
            else:
                adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                    self.config.DATA_NAME, self.config.SPLIT)
                save_path = os.path.join(self.config.ROOT,
                                         f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapters
            print("\n[NON-COMPOSITIONAL MODULE]: ")
            adapter_name = '{}_{}_{}_{}-GIEA'.format(self.config.MODEL_TYPE, 'non-compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # Add a flow layer for all loaded adapters
            print("\n[COMBINED MODULE]: ")
            # Add a flow layer for all loaded adapters
            print("==> Adding Flow module!")
            adapter_setup = Flow("compositional", "non-compositional")
            print(adapter_setup)
            if self.config.MODE == 'train':
                if self.config.CONTINUE_TRAIN:
                    save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                    print("==> Loading Adapter Flow from {}".format(save_path))
                    self.model.load_adapter_flow(save_path)
                    print("==>  Adapter Flow Loaded!")
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
                if os.path.exists(save_path):
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
        elif self.config.ADAPTER_NAME == 'fusion-cls':
            self.model.save_adapter_fusion(save_path, "compositional,non-compositional,disambiguate")
        else:
            self.model.save_adapter(save_path, self.config.ADAPTER_NAME)


class BartAdapterCombinedCLS(nn.Module):
    """
    PIER model for downstream classification tasks (e.g., sentiment classification or paraphrase identification)

    """
    def __init__(self, config):
        super(BartAdapterCombinedCLS, self).__init__()
        self.config = config
        self.vocab_size = BartConfig().vocab_size
        # Base BART Model
        print("Load base model from {}".format(config.PRETRAINED_MODEL_NAME))
        self.model = BartModelWithHeads.from_pretrained(config.PRETRAINED_MODEL_NAME)
        # Adapters
        adapter_config = AdapterConfig.load(config=config.ADAPTER_ARCHITECTURE_NAME,
                                            reduction_factor=config.ADAPTER_REDUCTION_FACTOR)

        self.attach_adapter(adapter_config)
        self.model.to(config.DEVICE)

    def forward(self, data, writer=None):
        outputs = self.model(**data['inputs'], labels=data['labels'])
        return outputs.loss, 0, outputs.logits

    def attach_adapter(self, adapter_config):
        if self.config.ADAPTER_NAME == 'fusion':
            print('=> Initializing Adapters with Fusion Module...')
            # load trained adapters
            # 1. load compositional adapters
            if self.config.USE_BART_AS_COMPOSITIONAL:
                print("\n[COMPOSITIONAL MODULE]: ")
                print('==> Using BART output as compositional embedding!')
                self.model.add_adapter("compositional", config='skip')
            else:
                adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                    self.config.DATA_NAME, self.config.SPLIT)
                save_path = os.path.join(self.config.ROOT,
                                         f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapters
            print("\n[NON-COMPOSITIONAL MODULE]: ")
            # Load GIEA's checkpoint as the non-compositional adapter
            adapter_name = '{}_{}_{}_{}-GIEA'.format(self.config.MODEL_TYPE, 'non-compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # Add a fusion layer for all loaded adapters
            print("\n[COMBINED MODULE]: ")
            print("==> Adding fusion module!")
            save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
            print("==> Loading Adapter Fusion from {}".format(save_path))
            self.model.load_adapter_fusion(save_path)
            fusion_adapter_setup = Fuse("compositional", "non-compositional")
            self.model.set_active_adapters(fusion_adapter_setup)

            return

        elif self.config.ADAPTER_NAME == 'flow':
            raise('not modified yet')
            # 1. load compositional adapter
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapter
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'non-compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # Add a flow layer for all loaded adapters
            print("==> Adding Flow module!")
            flow_adapter_setup = Flow("compositional", "non-compositional")
            print(flow_adapter_setup)
            # Train mode
            save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
            print("==> Loading Adapter Flow from {}".format(save_path))
            self.model.load_adapter_flow(save_path)

            return
        elif self.config.ADAPTER_NAME == 'non-compositional':
            adapter_name = '{}_{}_{}_{}-GIEA'.format(self.config.MODEL_TYPE, self.config.ADAPTER_NAME,
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            adapter_name = self.model.load_adapter(save_path, with_head=False)
            self.model.train_adapter(adapter_name)
        else:
            raise NotImplementedError('Adapter Name: {} is not valid!'.format(self.config.ADAPTER_NAME))

    def save_model(self, save_type):
        save_path = self.config.PATH_TO_CHECKPOINT.format(save_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.save_adapter(save_path, self.config.ADAPTER_NAME)


class BartAdapterCombinedBase(nn.Module):
    """
    PIER model with only reconstruction loss from copy objective
    """
    def __init__(self, config):
        super(BartAdapterCombinedBase, self).__init__()
        self.config = config
        self.vocab_size = BartConfig().vocab_size
        # Base BART Model
        print("Load base model from {}".format(config.PRETRAINED_MODEL_NAME))
        self.model = BartForConditionalGeneration.from_pretrained(config.PRETRAINED_MODEL_NAME)
        # Adapters
        adapter_config = AdapterConfig.load(config=config.ADAPTER_ARCHITECTURE_NAME,
                                            reduction_factor=config.ADAPTER_REDUCTION_FACTOR)
        # self.disambiguate_classifier = nn.Linear(config.PRETRAINED_EMBED_DIM, config.CLS_LI_NUM_CLASSES).to(config.DEVICE)
        self.attach_adapter(adapter_config)
        self.model.to(config.DEVICE)
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.nll_loss = nn.NLLLoss()

    def mean_pooling(self, token_embeddings, attention_mask):
        # token_embeddings: [batch_size, max_seq_len, hidden_dim]
        if token_embeddings.shape[1] != attention_mask.shape[1]:
            print('Shape mismatch occurred! Embed Shape: {} || Mask Shape: {}'.format(token_embeddings.shape, attention_mask.shape))
            attention_mask = torch.ones(token_embeddings.shape[0], token_embeddings.shape[1]).to(self.config.DEVICE)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compute_masked_ce_loss(self, logits, target, mask):
        """
        Args:
            logits: A Variable containing a FloatTensor of size
                (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            mask: The binary mask that indicates which tokens are counted
        Returns:
            loss: An average loss value masked by the length.
        """

        # logits_flat: (batch * max_len, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = F.log_softmax(logits_flat, -1)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # mask: (batch, max_len)
        mask *= self.config.LOSS_MASK_SCALE
        losses = losses * mask.float()
        length = torch.sum(mask, axis=1)
        loss = losses.sum() / length.float().sum()
        return loss

    def forward(self, data, writer=None):
        if self.config.MODE != 'inference':

            outputs = self.model(**data['inputs'], labels=data['labels'])
            # Reconstruction loss for infilling objectives
            l_recon = outputs.loss
            loss = l_recon
            return loss, loss, outputs.logits

        else:
            outputs = self.model.generate(**data, eos_token_id=1, num_beams=5, max_length=350)
            return outputs

    def attach_adapter(self, adapter_config):
        if self.config.ADAPTER_NAME == 'fusion-cls':
            print('=> Initializing Adapters with Fusion Module with CLS adapter...')
            # load trained adapters
            # 1. load compositional adapters
            if self.config.USE_BART_AS_COMPOSITIONAL:
                print("\n[COMPOSITIONAL MODULE]: ")
                print('==> Using BART output as compositional embedding!')
                self.model.add_adapter("compositional", config='skip')
            else:
                adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                    self.config.DATA_NAME, self.config.SPLIT)
                save_path = os.path.join(self.config.ROOT,
                                         f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapters
            print("\n[NON-COMPOSITIONAL MODULE]: ")
            adapter_name = '{}_{}_{}_{}-GIEA'.format(self.config.MODEL_TYPE, 'non-compositional',
                                                             self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # 3. load disambiguation adapter
            print("\n[DISAMBIGUATION MODULE]: ")
            adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'disambiguate',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)

            # Add a fusion layer for all loaded adapters
            print("\n[COMBINED MODULE]: ")
            print("==> Adding fusion module!")
            adapter_setup = Fuse("compositional", "non-compositional", "disambiguate")
            print(adapter_setup)
            if self.config.MODE == 'train':
                if self.config.CONTINUE_TRAIN:
                    save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                    print("==> Loading Adapter Fusion from {}".format(save_path))
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
                adapter_setup = Fuse("compositional", "non-compositional", "disambiguate")
                self.model.set_active_adapters(adapter_setup)
                # print("==> Loading Classification Head from {}".format(save_path))
                # self.model.load_head(save_path)
                self.model.train_adapter_fusion(adapter_setup)
                self.model.eval()

        elif self.config.ADAPTER_NAME == 'fusion':
            print('=> Initializing Adapters with Fusion Module...')
            # load trained adapters
            # 1. load compositional adapters
            if self.config.USE_BART_AS_COMPOSITIONAL:
                print("\n[COMPOSITIONAL MODULE]: ")
                print('==> Using BART output as compositional embedding!')
                self.model.add_adapter("compositional", config='skip')
            else:
                adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                    self.config.DATA_NAME, self.config.SPLIT)
                save_path = os.path.join(self.config.ROOT,
                                         f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapters
            print("\n[NON-COMPOSITIONAL MODULE]: ")
            adapter_name = '{}_{}_{}_{}-GIEA'.format(self.config.MODEL_TYPE, 'non-compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            print("==> Non-compositional adapter loaded!")
            # Add a fusion layer for all loaded adapters
            print("\n[COMBINED MODULE]: ")
            print("==> Adding fusion module!")
            adapter_setup = Fuse("compositional", "non-compositional")
            print(adapter_setup)
            if self.config.MODE == 'train':
                if self.config.CONTINUE_TRAIN:
                    save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                    print("==> Loading Adapter Fusion from {}".format(save_path))
                    self.model.load_adapter_fusion(save_path)
                    print("==>  Adapter Fusion Loaded!")
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

        elif 'flow' in self.config.ADAPTER_NAME:
            print('=> Initializing Adapters with <Flow> Module...')
            # load trained adapters
            # 1. load compositional adapters
            if self.config.USE_BART_AS_COMPOSITIONAL:
                print("\n[COMPOSITIONAL MODULE]: ")
                print('==> Using BART output as compositional embedding!')
                self.model.add_adapter("compositional", config='skip')
            else:
                adapter_name = '{}_{}_{}_{}'.format(self.config.MODEL_TYPE, 'compositional',
                                                    self.config.DATA_NAME, self.config.SPLIT)
                save_path = os.path.join(self.config.ROOT,
                                         f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=False)
            # 2. load non-compositional adapters
            print("\n[NON-COMPOSITIONAL MODULE]: ")
            adapter_name = '{}_{}_{}_{}-GIEA'.format(self.config.MODEL_TYPE, 'non-compositional',
                                                self.config.DATA_NAME, self.config.SPLIT)
            save_path = os.path.join(self.config.ROOT,
                                     f'checkpoints/{adapter_name}/{self.config.LOAD_CHECKPOINT_TYPE}/')
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=False)
            # Add a flow layer for all loaded adapters
            print("\n[COMBINED MODULE]: ")
            # Add a flow layer for all loaded adapters
            print("==> Adding Flow module!")
            adapter_setup = Flow("compositional", "non-compositional")
            print(adapter_setup)
            if self.config.MODE == 'train':
                if self.config.CONTINUE_TRAIN:
                    save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
                    print("==> Loading Adapter Flow from {}".format(save_path))
                    self.model.load_adapter_flow(save_path)
                    print("==>  Adapter Flow Loaded!")
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
                adapter_name = '{}_{}_{}_{}-GIEA'.format(self.config.MODEL_TYPE, self.config.ADAPTER_NAME,
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
                if os.path.exists(save_path):
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
        elif self.config.ADAPTER_NAME == 'fusion-cls':
            self.model.save_adapter_fusion(save_path, "compositional,non-compositional,disambiguate")
        else:
            self.model.save_adapter(save_path, self.config.ADAPTER_NAME)
