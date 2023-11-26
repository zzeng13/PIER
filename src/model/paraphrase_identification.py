import os
from adapter_variants.src.transformers.models.bert import BertModelWithHeads, BertConfig
from adapter_variants.src.transformers.models.bart import BartModelWithHeads, BartConfig
from adapter_variants.src.transformers.adapters.composition import Fuse, Flow, Stack
from adapter_variants.src.transformers.adapters.configuration import AdapterConfig
from torch import nn


class ParaphraseIdentificationAdapter(nn.Module):
    def __init__(self, config):
        super(ParaphraseIdentificationAdapter, self).__init__()
        self.config = config
        # Base BART Model
        # Load flow module
        self.model = BartModelWithHeads.from_pretrained(self.config.PRETRAINED_MODEL_NAME, num_labels=2)
        # Adapters
        adapter_config = AdapterConfig.load(config=self.config.ADAPTER_ARCHITECTURE_NAME,
                                            reduction_factor=self.config.ADAPTER_REDUCTION_FACTOR)

        self.attach_adapter(adapter_config)
        self.model.to(config.DEVICE)

    def forward(self, data):

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
            # self.model.train_adapter_fusion(adapter_setup)
            if self.config.MODE == 'train':
                # Load a classification adapter and classification head
                adapter_name = 'paraphrase_identification'
                pi_adapter_setup = self.model.add_adapter(adapter_name, config=adapter_config)
                # specify the adapter to train
                self.model.add_classification_head(adapter_name, num_labels=2)
                self.model.train_adapter(adapter_name)
                self.model.active_adapters = Stack(fusion_adapter_setup, adapter_name)
                print('[ACTIVATE ADAPTERS]: ')
                print(self.model.active_adapters)
                print('------------------------------------------------')
                print('[TRAINABLE ADAPTERS]: ')
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name)
                print('------------------------------------------------')
            else:
                adapter_name = 'paraphrase_identification'
                save_path = os.path.join(self.config.ROOT, 'checkpoints/paraphrase_identification/{}/{}/{}/'.format(
                    self.config.ADAPTER_NAME,
                    self.config.LOAD_CHECKPOINT_TYPE,
                    '-'.join(self.config.TRAIN_DATA_NAMES)))
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=True)
                self.model.active_adapters = Stack(fusion_adapter_setup, adapter_name)
                print(self.model.active_adapters)
                self.model.eval()
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
            if self.config.MODE == 'train':
                # Load a classification adapter and classification head
                adapter_name = 'paraphrase_identification'
                pi_adapter_setup = self.model.add_adapter(adapter_name, config=adapter_config)
                # specify the adapter to train
                self.model.add_classification_head(adapter_name, num_labels=2)
                self.model.train_adapter(adapter_name)

                self.model.active_adapters = Stack(flow_adapter_setup, adapter_name)
                print('[ACTIVATE ADAPTERS]: ')
                print(self.model.active_adapters)
                print('------------------------------------------------')
                print('[TRAINABLE ADAPTERS]: ')
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name)
                print('------------------------------------------------')
            else:
                adapter_name = 'paraphrase_identification'
                save_path = os.path.join(self.config.ROOT, 'checkpoints/paraphrase_identification/{}/{}/{}/'.format(
                    self.config.ADAPTER_NAME,
                    self.config.LOAD_CHECKPOINT_TYPE,
                    '-'.join(self.config.TRAIN_DATA_NAMES)))
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=True)
                self.model.active_adapters = Stack(flow_adapter_setup, adapter_name)
                print(self.model.active_adapters)
                self.model.eval()
            return
        elif self.config.ADAPTER_NAME == 'single':
            if self.config.MODE == 'train':
                # Load a classification adapter and classification head
                adapter_name = 'paraphrase_identification'
                pi_adapter_setup = self.model.add_adapter(adapter_name, config=adapter_config)
                self.model.add_classification_head(adapter_name, num_labels=2)
                # specify the adapter to train
                self.model.set_active_adapters(adapter_name)
                self.model.train_adapter(adapter_name)
                print(self.model.active_adapters)
                print('------------------------------------------------')
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name)
                print('------------------------------------------------')
            else:
                adapter_name = 'paraphrase_identification'
                save_path = os.path.join(self.config.ROOT, 'checkpoints/paraphrase_identification/{}/{}/{}/'.format(
                    self.config.ADAPTER_NAME,
                    self.config.LOAD_CHECKPOINT_TYPE,
                    '-'.join(self.config.TRAIN_DATA_NAMES)))
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=True)
                self.model.set_active_adapters(adapter_name)
                print(self.model.active_adapters)
                self.model.eval()

        else:
            raise NotImplementedError('Adapter Name: {} is not valid!'.format(self.config.ADAPTER_NAME))

    def save_model(self, save_type):
        save_path = os.path.join(self.config.ROOT, 'checkpoints/paraphrase_identification/{}/{}/{}/'.format(
            self.config.ADAPTER_NAME,
            save_type,
            '-'.join(self.config.TRAIN_DATA_NAMES)))
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.save_adapter(save_path, 'paraphrase_identification', with_head=True)


class ParaphraseIdentificationAdapterBERT(nn.Module):
    def __init__(self, config):
        super(ParaphraseIdentificationAdapterBERT, self).__init__()
        self.config = config
        # Base BART Model
        # Load flow module
        self.model = BertModelWithHeads.from_pretrained(self.config.PRETRAINED_MODEL_NAME, num_labels=2)
        # Adapters
        adapter_config = AdapterConfig.load(config=self.config.ADAPTER_ARCHITECTURE_NAME,
                                            reduction_factor=self.config.ADAPTER_REDUCTION_FACTOR)

        self.attach_adapter(adapter_config)
        self.model.to(config.DEVICE)

    def forward(self, data):

        outputs = self.model(**data['inputs'], labels=data['labels'])
        return outputs.loss, outputs.logits

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
            save_path = self.config.PATH_TO_CHECKPOINT.format(self.config.LOAD_CHECKPOINT_TYPE)
            print("==> Loading Adapter Fusion from {}".format(save_path))
            self.model.load_adapter_fusion(save_path)
            fusion_adapter_setup = Fuse("compositional", "non-compositional")
            self.model.set_active_adapters(adapter_setup)
            # print("==> Loading Classification Head from {}".format(save_path))
            # self.model.load_head(save_path)
            self.model.train_adapter_fusion(adapter_setup)
            self.model.eval()

            if self.config.MODE == 'train':
                # Load a classification adapter and classification head
                adapter_name = 'paraphrase_identification'
                pi_adapter_setup = self.model.add_adapter(adapter_name, config=adapter_config)
                # specify the adapter to train
                self.model.add_classification_head(adapter_name, num_labels=2)
                self.model.train_adapter(adapter_name)

                self.model.active_adapters = Stack(fusion_adapter_setup, adapter_name)
                print('[ACTIVATE ADAPTERS]: ')
                print(self.model.active_adapters)
                print('------------------------------------------------')
                print('[TRAINABLE ADAPTERS]: ')
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name)
                print('------------------------------------------------')
            else:
                adapter_name = 'paraphrase_identification'
                save_path = os.path.join(self.config.ROOT, 'checkpoints/paraphrase_identification/{}/{}/{}/'.format(
                    self.config.ADAPTER_NAME,
                    self.config.LOAD_CHECKPOINT_TYPE,
                    '-'.join(self.config.TRAIN_DATA_NAMES)))
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=True)
                self.model.active_adapters = Stack(fusion_adapter_setup, adapter_name)
                print(self.model.active_adapters)
                self.model.eval()
                print('[ACTIVATE ADAPTERS]: ')
                print(self.model.active_adapters)
                print('------------------------------------------------')
                print('[TRAINABLE ADAPTERS]: ')
                for name, param in self.model.named_parameters():
                    param.requires_grad = True
                    if param.requires_grad:
                        print(name)
                print('------------------------------------------------')
            return


        elif self.config.ADAPTER_NAME == 'flow':
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
            if self.config.MODE == 'train':
                # Load a classification adapter and classification head
                adapter_name = 'paraphrase_identification'
                pi_adapter_setup = self.model.add_adapter(adapter_name, config=adapter_config)
                # specify the adapter to train
                self.model.add_classification_head(adapter_name, num_labels=2)
                self.model.train_adapter(adapter_name)

                self.model.active_adapters = Stack(flow_adapter_setup, adapter_name)
                print('[ACTIVATE ADAPTERS]: ')
                print(self.model.active_adapters)
                print('------------------------------------------------')
                print('[TRAINABLE ADAPTERS]: ')
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name)
                print('------------------------------------------------')
            else:
                adapter_name = 'paraphrase_identification'
                save_path = os.path.join(self.config.ROOT, 'checkpoints/paraphrase_identification/{}/{}/{}/'.format(
                    self.config.ADAPTER_NAME,
                    self.config.LOAD_CHECKPOINT_TYPE,
                    '-'.join(self.config.TRAIN_DATA_NAMES)))
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=True)
                self.model.active_adapters = Stack(flow_adapter_setup, adapter_name)
                print(self.model.active_adapters)
                self.model.eval()
                print('[ACTIVATE ADAPTERS]: ')
                print(self.model.active_adapters)
                print('------------------------------------------------')
                print('[TRAINABLE ADAPTERS]: ')
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name)
                print('------------------------------------------------')

            return
        elif self.config.ADAPTER_NAME == 'single':
            if self.config.MODE == 'train':
                # Load a classification adapter and classification head
                adapter_name = 'paraphrase_identification'
                pi_adapter_setup = self.model.add_adapter(adapter_name, config=adapter_config)
                self.model.add_classification_head(adapter_name, num_labels=2)
                # specify the adapter to train
                self.model.set_active_adapters(adapter_name)
                self.model.train_adapter(adapter_name)
                print(self.model.active_adapters)
                print('------------------------------------------------')
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name)
                print('------------------------------------------------')
            else:
                adapter_name = 'paraphrase_identification'
                save_path = os.path.join(self.config.ROOT, 'checkpoints/paraphrase_identification/{}/{}/{}/'.format(
                    self.config.ADAPTER_NAME,
                    self.config.LOAD_CHECKPOINT_TYPE,
                    '-'.join(self.config.TRAIN_DATA_NAMES)))
                print("==> Loading Adapter from {}".format(save_path))
                self.model.load_adapter(save_path, with_head=True)
                self.model.set_active_adapters(adapter_name)
                print(self.model.active_adapters)
                self.model.eval()
                print('[ACTIVATE ADAPTERS]: ')
                print(self.model.active_adapters)
                print('------------------------------------------------')
                print('[TRAINABLE ADAPTERS]: ')
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name)
                print('------------------------------------------------')

        else:
            raise NotImplementedError('Adapter Name: {} is not valid!'.format(self.config.ADAPTER_NAME))

    def save_model(self, save_type):
        save_path = os.path.join(self.config.ROOT, 'checkpoints/paraphrase_identification/{}/{}/{}/'.format(
            self.config.ADAPTER_NAME,
            save_type,
            '-'.join(self.config.TRAIN_DATA_NAMES)))
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.save_adapter(save_path, 'paraphrase_identification', with_head=True)
