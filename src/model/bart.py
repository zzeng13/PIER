from adapter_variants.src.transformers.models.bart import BartForConditionalGeneration, BartConfig
from torch import nn


class BartModel(nn.Module):
    def __init__(self, config):
        super(BartModel, self).__init__()
        self.config = config
        self.vocab_size = BartConfig().vocab_size
        # Base BART Model
        self.model = BartForConditionalGeneration.from_pretrained(config.PRETRAINED_MODEL_NAME)

        self.model.to(config.DEVICE)

    def forward(self, data):
        if self.config.MODE == 'train':
            outputs = self.model(**data['inputs'], labels=data['labels'])
            return outputs.loss, outputs.logits

        else:
            outputs = self.model.generate(**data['inputs'], num_beams=5, max_length=150)
            return outputs

    def save_model(self, save_type):
        save_path = self.config.PATH_TO_SAVE_PRETRAINED.format(save_type)
        self.model.save_pretrained(save_path)




