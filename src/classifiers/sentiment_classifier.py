import os
from torch import nn
from src.model.bart_adapters import BartAdapterCLS, BartAdapter, BartAdapterCombined, BartAdapterCombinedCLS
from adapter_variants.src.transformers.adapters.configuration import AdapterConfig
from adapter_variants.src.transformers.models.bart import BartModelWithHeads, BartConfig, BartModel
from adapter_variants.src.transformers.adapters.composition import Fuse, Flow, Stack


class SentimentClassifier(nn.Module):
    def __init__(self, config):
        super(SentimentClassifier, self).__init__()
        # Adapter embedding layer
        self.config = config
        # Adapters
        if self.config.ADAPTER_NAME == 'bart':
            self.model = BartModelWithHeads.from_pretrained('./bart-base')

        else:
            self.model = BartAdapterCombinedCLS(config).model
            # Print information
            print('[ACTIVATE TRAINED ADAPTERS]: ')
            print(self.model.active_adapters)

        # Adapter for classification
        adapter_config = AdapterConfig.load(config=self.config.ADAPTER_ARCHITECTURE_NAME,
                                            reduction_factor=self.config.ADAPTER_REDUCTION_FACTOR)
        if self.config.MODE == 'train':
            # Load a classification adapter and classification head
            adapter_name = 'sentiment_classification'
            pi_adapter_setup = self.model.add_adapter(adapter_name, config=adapter_config)
            # specify the adapter to train
            self.model.add_classification_head(adapter_name, num_labels=2)
            self.model.train_adapter(adapter_name)
            print('[ACTIVATE ADAPTERS]: ')
            if self.config.ADAPTER_NAME == 'fusion':
                self.model.active_adapters = Stack(Fuse("compositional", "non-compositional"), adapter_name)
            elif self.config.ADAPTER_NAME == 'flow':
                self.model.active_adapters = Stack(Flow("compositional", "non-compositional"), adapter_name)
            elif self.config.ADAPTER_NAME != 'bart':
                self.model.active_adapters = Stack(self.config.ADAPTER_NAME, adapter_name)
            else:
                self.model.active_adapters = Stack(adapter_name)
            print(self.model.active_adapters)
        else:
            adapter_name = 'sentiment_classification'
            save_path = self.config.PATH_TO_CHECKPOINT_SC.format(self.config.LOAD_CHECKPOINT_TYPE)
            print("==> Loading Adapter from {}".format(save_path))
            self.model.load_adapter(save_path, with_head=True)
            print("==> Successfully loaded Adapter from {}".format(save_path))
            if self.config.ADAPTER_NAME == 'fusion':
                self.model.active_adapters = Stack(Fuse("compositional", "non-compositional"), adapter_name)
            elif self.config.ADAPTER_NAME == 'flow':
                self.model.active_adapters = Stack(Flow("compositional", "non-compositional"), adapter_name)
            elif self.config.ADAPTER_NAME != 'bart':
                self.model.active_adapters = Stack(self.config.ADAPTER_NAME, adapter_name)
            else:
                self.model.active_adapters = Stack(adapter_name)

            print(self.model.active_adapters)
            self.model.eval()
        self.nll_loss = nn.NLLLoss()

    def save_model(self, save_type):
        save_path = self.config.PATH_TO_CHECKPOINT_SC.format(save_type)
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.save_adapter(save_path, 'sentiment_classification', with_head=True)

    def forward(self, data):
        outputs = self.model(**data['inputs'], labels=data['labels'])
        return outputs.loss, outputs.logits




