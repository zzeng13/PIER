"""
Implement necessary components for an encoder-decoder framework

- The encoder-decoder framework is specifically design for the inference in this project and
  hence not for general seq2seq tasks
- Key things:
    - the encoder has no embedding layer. The input to the encoder is already considered as 'embeded' hidden
    representations.
    - the decoder has its own embedding layer for the output classes, and it is not shared with other embeddings.

"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from src_idiom_detect.model.pretrained_embedding import EembeddingGeneratorAdapterBart

class Seq2SeqBERT(nn.Module):
    def __init__(self, config):
        super(Seq2SeqBERT, self).__init__()
        self.config = config
        self.teacher_forcing_ratio = config.BILSTM_TEACHER_FORCING_RATIO
        self.embedding_layer = EembeddingGeneratorAdapterBart(config)
        self.embedding_transform_layer = nn.Linear(config.PRETRAINED_EMBED_DIM, config.TGT_VOCAB_SIZE, bias=True)

    def forward(self, xs):  # context lens
        # encoder_output: batch_size, max_xs_seq_len, en_hidden_size
        xs = self.embedding_layer(xs)
        # encoder_output: batch_size, max_xs_seq_len, num_classes
        xs = self.embedding_transform_layer(xs)

        return F.log_softmax(xs, dim=-1)


"""
Multi-layer perceptron model.
"""

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            # nn.BatchNorm1d(input_dim // 2),  # applying batch norm
            nn.Dropout(dropout_rate),  # Note: dropout applied before RELU
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            # nn.BatchNorm1d(input_dim // 4),  # applying batch norm
            nn.Dropout(dropout_rate),  # Note: dropout applied before RELU
            nn.ReLU(),
            nn.Linear(input_dim // 4, output_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class Seq2SeqBiLSTMLite(nn.Module):
    def __init__(self, config):
        super(Seq2SeqBiLSTMLite, self).__init__()
        self.config = config
        self.src_embedding_layer = EembeddingGeneratorAdapterBart(config)
        self.src_embedding_transform_layer = nn.Linear(config.PRETRAINED_EMBED_DIM, config.EMBEDDING_DIM)
        # self.encoder = EncoderBiLSTM(config)
        self.output_linear = MultiLayerPerceptron(config.EMBEDDING_DIM, config.TGT_VOCAB_SIZE)

    def forward(self, xs, x_lens, ys, training=False):  # context lens
        xs = self.src_embedding_layer(xs)
        xs = self.src_embedding_transform_layer(xs)
        batch_size = xs.shape[0]
        max_ys_seq_len = ys.shape[1] - 1  # minus the start symbol
        # xs: batch_size, max_xs_seq_len, en_hidden_size
        # decoder_hidden: (h, c)
        # h, c: 1, batch_size, en_hidden_size
        # xs, _ = self.encoder(xs, x_lens)
        # batch_size, max_ys_seq_len, vocab_size
        xs = self.output_linear(xs)

        if training:
            return F.log_softmax(xs, dim=-1)
        else:
            return F.log_softmax(xs, dim=-1), None


class Seq2SeqBiLSTM(nn.Module):
    def __init__(self, config):
        super(Seq2SeqBiLSTM, self).__init__()
        self.config = config
        self.teacher_forcing_ratio = config.BILSTM_TEACHER_FORCING_RATIO
        self.src_embedding_layer = EembeddingGeneratorAdapterBart(config)
        self.src_embedding_transform_layer = nn.Linear(config.PRETRAINED_EMBED_DIM, config.EMBEDDING_DIM)
        self.encoder = EncoderBiLSTM(config)
        self.decoder = DecoderLSTMWithAttention(config)

    def forward(self, xs, x_lens, ys, training=False):  # context lens
        xs = self.src_embedding_layer(xs)
        xs = self.src_embedding_transform_layer(xs)
        batch_size = xs.shape[0]
        max_ys_seq_len = ys.shape[1] - 1  # minus the start symbol
        # encoder_output: batch_size, max_xs_seq_len, en_hidden_size
        # decoder_hidden: (h, c)
        # h, c: 1, batch_size, en_hidden_size
        encoder_output, decoder_hidden = self.encoder(xs, x_lens)
        # batch_size
        decoder_input = torch.empty(
            batch_size,
            dtype=torch.int64,
            device=self.config.DEVICE)
        decoder_input.fill_(self.config.START_IDX)
        # max_ys_seq_len, batch_size, vocab_size
        decoder_outputs = torch.zeros(
            max_ys_seq_len,
            batch_size,
            self.config.TGT_VOCAB_SIZE,
            device=self.config.DEVICE)
        if not training:
            attn_scores = torch.zeros(
                batch_size,
                max_ys_seq_len+1,
                max_ys_seq_len+1,
                device=self.config.DEVICE)

        # greedy search with teacher forcing
        for i in range(max_ys_seq_len):
            # decoder_output: batch_size, vocab_size
            # decoder_hidden: (h, c)
            # h, c: 1, batch_size, en_hidden_size
            decoder_output, decoder_hidden, attn_w = self.decoder(
                decoder_input, decoder_hidden, encoder_output, x_lens)
            # batch_size, vocab_size
            decoder_outputs[i] = decoder_output
            # print(attn_w.shape)
            if not training:
                attn_scores[:, i, :] = attn_w.squeeze(1)
            # batch_size ys needs to skip the start symbol
            decoder_input = ys[:, i + 1] if random.random() < self.teacher_forcing_ratio and training \
                else decoder_output.max(1)[1]
        # batch_size, max_ys_seq_len, vocab_size
        if training:
            return decoder_outputs.transpose(0, 1)
        else:
            return decoder_outputs.transpose(0, 1), attn_scores


# Bi-LSTM based Encoder
class EncoderBiLSTM(nn.Module):
    def __init__(self, config):
        super(EncoderBiLSTM, self).__init__()
        self.config = config
        # LSTM model
        self.bilstm = nn.LSTM(
            input_size=self.config.EMBEDDING_DIM,
            hidden_size=self.config.ENCODER_HIDDEN_DIM // 2,
            num_layers=self.config.BILSTM_ENCODER_NUM_LAYERS,
            batch_first=True,
            dropout=self.config.BILSTM_ENCODER_DROP_RATE,
            bidirectional=True
        )
        # Dropout
        self.lstm_dropout_layer = nn.Dropout(self.config.BILSTM_ENCODER_DROP_RATE)

    def forward(self, x, x_lens):
        x = nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=x_lens.to('cpu'),
            batch_first=True,
            enforce_sorted=False)
        x, (h, c) = self.bilstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=x,
            batch_first=True)

        # shape:
        h = torch.unsqueeze(torch.cat(torch.unbind(h, 0), 1), 0)
        # shape:
        c = torch.unsqueeze(torch.cat(torch.unbind(c, 0), 1), 0)
        # shape:
        x = self.lstm_dropout_layer(x)

        return x, (h, c)


# Attention Layer (for Decoder)
class Seq2SeqAttention(nn.Module):
    def __init__(self, config):
        super(Seq2SeqAttention, self).__init__()
        self.config = config
        # Attention Layer
        self.attn_layer = torch.nn.Linear(
            2 * self.config.ENCODER_HIDDEN_DIM + self.config.DECODER_HIDDEN_DIM,
            self.config.DECODER_HIDDEN_DIM
        )
        self.v = torch.nn.Parameter(torch.rand(self.config.DECODER_HIDDEN_DIM))
        self.v.data.normal_(mean=0, std=1./self.v.size(0)**(1./2.))

    def compute_attn_score(self, hidden_states, encoder_output):
        # hidden: batch_size, max_src_seq_len, de_hidden_size*2
        # encoder_output: batch_size, max_src_seq_len, en_hidden_size
        # batch_size, max_src_seq_len, de_hidden_size
        energy = torch.tanh(self.attn_layer(torch.cat([hidden_states, encoder_output], 2)))
        # batch_size, de_hidden_size, max_src_seq_len
        energy = energy.transpose(2, 1)
        # batch_size, 1, de_hidden_size
        v = self.v.repeat(encoder_output.shape[0], 1).unsqueeze(1)
        # batch_size, 1, max_src_seq_len
        energy = torch.bmm(v, energy)
        # batch_size, max_src_seq_len
        return energy.squeeze(1)

    def forward(self, hidden_states, encoder_output, src_seq_lens):
        # hidden: (h, c)
        # h, c: 1, batch_size, de_hidden_size
        # encoder_output: batch_size, max_src_seq_len, en_hidden_size
        # src_lens: batch_size
        # 1, batch_size, de_hidden_size*2
        hidden_states = torch.cat(hidden_states, 2).squeeze(0)
        # batch_size, max_src_seq_len, de_hidden_size*2
        hidden_states = hidden_states.repeat(encoder_output.shape[1], 1, 1).transpose(0, 1)
        # batch_size, max_src_seq_len
        attn_energies = self.compute_attn_score(hidden_states, encoder_output)
        # max_src_seq_len
        idx = torch.arange(end=encoder_output.shape[1], dtype=torch.float, device=self.config.DEVICE)
        # batch_size, max_src_seq_len
        idx = idx.unsqueeze(0).expand(attn_energies.shape)
        # batch size, max_src_seq_len
        src_lens = src_seq_lens.unsqueeze(-1).expand(attn_energies.shape)
        mask = idx.long() < src_lens
        attn_energies[~mask] = float('-inf')
        # batch_size, 1, max_src_seq_len
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# LSTM based Decoder
class DecoderLSTMWithAttention(nn.Module):
    def __init__(self, config):
        super(DecoderLSTMWithAttention, self).__init__()
        self.config = config
        # Embedding layer
        self.embedding_layer = nn.Embedding(
            num_embeddings=self.config.TGT_VOCAB_SIZE,
            embedding_dim=self.config.EMBEDDING_DIM,  # Note: this could be different from LSTM hidden dim
            padding_idx=self.config.PAD_IDX
        )
        self.embedding_dropout_layer = nn.Dropout(self.config.BILSTM_LINEAR_DROP_RATE)
        # Attention layer
        self.attn_layer = Seq2SeqAttention(self.config)
        self.atten_combine_layer = torch.nn.Linear(
            self.config.EMBEDDING_DIM + self.config.ENCODER_HIDDEN_DIM,
            self.config.DECODER_HIDDEN_DIM
        )
        # LSTM model
        self.lstm_model = nn.LSTM(
            input_size=self.config.EMBEDDING_DIM,
            hidden_size=self.config.DECODER_HIDDEN_DIM,
            num_layers=self.config.BILSTM_DECODER_NUM_LAYERS,
            batch_first=True,
            dropout=0,
            bidirectional=False)
        self.lstm_dropout_layer = nn.Dropout(self.config.BILSTM_DECODER_DROP_RATE)
        # Output layer
        self.output_layer = torch.nn.Linear(self.config.DECODER_HIDDEN_DIM, self.config.TGT_VOCAB_SIZE)

    def forward(self, x, hidden_states, encoder_output, src_seq_lens):
        # x: batch_size
        # hidden: (h, c)
        # h, c: 1, batch_size, de_hidden_size
        # encoder_output: batch_size, max_src_seq_len, en_hidden_size
        # src_lens: batch_size
        # batch_size, 1, embedding_dim
        x = self.embedding_layer(x).unsqueeze(1)
        x = self.embedding_dropout_layer(x)
        # batch_size, 1, max_src_seq_len
        attn_w = self.attn_layer(hidden_states, encoder_output, src_seq_lens)
        # batch_size, 1, en_hidden_size
        context = attn_w.bmm(encoder_output)
        # batch_size, 1, de_hidden_size
        x = self.atten_combine_layer(torch.cat((x, context), 2))
        x = F.relu(x)
        x, (h, c) = self.lstm_model(x, hidden_states)
        # batch_size, de_hidden_size
        x = self.lstm_dropout_layer(x).squeeze(1)
        # batch_size, tgt_vocab_size
        x = self.output_layer(x)
        return F.log_softmax(x, dim=-1), (h, c), attn_w



