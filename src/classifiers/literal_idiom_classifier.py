from torch import nn
import torch.nn.functional as F
from src.model.bart_adapters import BartAdapter, BartAdapterCombined
from src.model.bert_adapters import BertAdapter
from transformers import BartModel, BertForMaskedLM

from src.utils.model_util import *


class LiteralIdiomaticClassifier(nn.Module):
    def __init__(self, config):
        super(LiteralIdiomaticClassifier, self).__init__()
        # Adapter embedding layer
        self.config = config
        # Adapters

        if self.config.ADAPTER_NAME == 'bart':
            self.adapter = BartModel.from_pretrained('./bart-base')

        elif self.config.ADAPTER_NAME == 'bert':
            self.adapter = BertForMaskedLM.from_pretrained(config.PRETRAINED_BERT_NAME, output_hidden_states=True)

        else:
            if self.config.USE_COMBINED:
                # Load the PIER model
                self.adapter = BartAdapterCombined(config)
            else:
                if 'bert' in self.config.MODEL_TYPE:
                    self.adapter = BertAdapter(config)
                    print('Loaded BERT adapter!')
                else:
                    self.adapter = BartAdapter(config)
            # Print information
            print('[ACTIVATE ADAPTERS]: ')
            print(self.adapter.model.active_adapters)
        # Simple projection/linear layer
        self.projection_layer = nn.Linear(
            self.config.PRETRAINED_HIDDEN_SIZE,
            self.config.CLS_LI_NUM_CLASSES
        )
        self.nll_loss = nn.NLLLoss()

    def adapter_embedding(self, data):
        if self.config.ADAPTER_NAME in ['bart', 'bert']:
            outputs = self.adapter(**data, output_hidden_states=True)
        else:
            outputs = self.adapter.model(**data, output_hidden_states=True)

        if self.config.LAYER_TYPE == 'last':
            if 'bert' in self.config.MODEL_TYPE:
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs.last_hidden_state
        elif self.config.LAYER_TYPE == 'all':
            dec_hidden_states = torch.stack(outputs.decoder_hidden_states)
            enc_hidden_states = torch.stack(outputs.encoder_hidden_states)
            hidden_states = torch.mean(torch.vstack([enc_hidden_states, dec_hidden_states]), 0)
        elif self.config.LAYER_TYPE == 'decoder-only':
            hidden_states = torch.mean(torch.stack(outputs.decoder_hidden_states), 0)
        return hidden_states

    def mean_pooling(self, token_embeddings, attention_mask):
        # token_embeddings: [batch_size, max_seq_len, hidden_dim]
        # token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, data):
        # 1. get embeddings from adapter
        x = self.adapter_embedding(data['inputs'])
        # 2. mean pooling the phrases
        x = self.mean_pooling(x, data['phrase_masks'])
        # 3. projection layers
        x = self.projection_layer(x)
        logits = F.log_softmax(x, dim=-1)
        loss = self.nll_loss(logits, data['labels'])
        return loss, logits

    def encode(self, data):
        # 1. get embeddings from adapter
        x = self.adapter_embedding(data['inputs'])
        # 2. mean pooling the phrases
        x = self.mean_pooling(x, data['phrase_masks'])
        return x








