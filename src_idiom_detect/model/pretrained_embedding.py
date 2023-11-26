from torch import nn
from src.model.bart_adapters import BartAdapter, BartAdapterCombined
from src.model.bert_adapters import BertAdapter

from transformers import BartModel, BertModel, BertForMaskedLM
from src.utils.model_util import *


class EembeddingGeneratorAdapterBart(nn.Module):
    def __init__(self, config):
        super(EembeddingGeneratorAdapterBart, self).__init__()
        self.config = config
        if self.config.ADAPTER_NAME == 'bart':
            self.adapter = BartModel.from_pretrained('./bart-base')

        elif self.config.ADAPTER_NAME == 'bart-ft':
            self.adapter, _, _ = load_init_model(BartAdapter, config)
        elif self.config.ADAPTER_NAME == 'bert':
            self.adapter = BertForMaskedLM.from_pretrained(config.PRETRAINED_BERT_NAME, output_hidden_states=True)

        else:
            if self.config.USE_COMBINED:
                self.adapter = BartAdapterCombined(config)
            else:
                if 'bert' in self.config.MODEL_TYPE:
                    self.adapter = BertAdapter(config)
                    print('Loaded BERT adapter!')
                else:
                    self.adapter = BartAdapter(config)

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

    def forward(self, data):
        x = self.adapter_embedding(data)
        return x