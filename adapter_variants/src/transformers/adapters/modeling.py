import math

import torch
from torch import nn


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):

        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":

            def swish(x):
                return x * torch.sigmoid(x)

            self.f = swish
        elif hidden_act.lower() == "gelu":

            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu

        super().__init__()

    def forward(self, x):
        return self.f(x)


# Single Adapter

class Adapter(nn.Module):
    """
    Implementation of a single Adapter block.
    """

    def __init__(
            self,
            input_size,
            down_sample=None,
            non_linearity="relu",
            init_bert_weights=True,
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            residual_before_ln=True,
            skip=False

    ):
        super().__init__()
        self.skip = skip
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual_before_ln = residual_before_ln
        if not self.skip:
            # list for all modules of the adapter, passed into nn.Sequential()
            seq_list = []

            # If we want to have a layer norm on input, we add it to seq_list
            if self.add_layer_norm_before:
                self.adapter_norm_before = nn.LayerNorm(self.input_size)
                seq_list.append(self.adapter_norm_before)

            # if a downsample size is not passed, we just half the size of the original input
            self.down_sample = down_sample
            if down_sample is None:
                self.down_sample = self.input_size // 2

            # Linear down projection of the input
            seq_list.append(nn.Linear(self.input_size, self.down_sample))

            # select non-linearity
            self.non_linearity = Activation_Function_Class(non_linearity.lower())

            seq_list.append(self.non_linearity)

            # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
            # residual connection
            self.adapter_down = nn.Sequential(*seq_list)

            # Up projection to input size
            self.adapter_up = nn.Linear(self.down_sample, self.input_size)

            # If we want to have a layer norm on output, we apply it later after a separate residual connection
            # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
            if self.add_layer_norm_after:
                self.adapter_norm_after = nn.LayerNorm(self.input_size)

            # if we want to initialize with the bert strategy then this function is called for all the linear layers
            if init_bert_weights:
                self.adapter_down.apply(self.init_bert_weights)
                self.adapter_up.apply(self.init_bert_weights)

    def forward(self, x, residual_input):  # , residual_input=None):
        if not self.skip:
            down = self.adapter_down(x)

            up = self.adapter_up(down)

            output = up

            # apply residual connection before layer norm if configured in this way
            if self.residual_before_ln:
                output = output + residual_input

            # apply layer norm if available
            if self.add_layer_norm_after:
                output = self.adapter_norm_after(output)

            # if residual should be applied after layer norm, apply it here
            if not self.residual_before_ln:
                output = output + residual_input

            return output, self.skip, up
        else:
            return x, self.skip, x

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


# class Adapter(nn.Module):
#     """
#     Implementation of a single Adapter block.
#     """
#
#     def __init__(
#         self,
#         input_size,
#         down_sample=None,
#         non_linearity="relu",
#         init_bert_weights=True,
#         add_layer_norm_before=True,
#         add_layer_norm_after=False,
#         residual_before_ln=True,
#
#     ):
#         super().__init__()
#
#         self.input_size = input_size
#         self.add_layer_norm_before = add_layer_norm_before
#         self.add_layer_norm_after = add_layer_norm_after
#         self.residual_before_ln = residual_before_ln
#
#         # list for all modules of the adapter, passed into nn.Sequential()
#         seq_list = []
#
#         # If we want to have a layer norm on input, we add it to seq_list
#         if self.add_layer_norm_before:
#             self.adapter_norm_before = nn.LayerNorm(self.input_size)
#             seq_list.append(self.adapter_norm_before)
#
#         # if a downsample size is not passed, we just half the size of the original input
#         self.down_sample = down_sample
#         if down_sample is None:
#             self.down_sample = self.input_size // 2
#
#         # Linear down projection of the input
#         seq_list.append(nn.Linear(self.input_size, self.down_sample))
#
#         # select non-linearity
#         self.non_linearity = Activation_Function_Class(non_linearity.lower())
#
#         seq_list.append(self.non_linearity)
#
#         # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
#         # residual connection
#         self.adapter_down = nn.Sequential(*seq_list)
#
#         # Up projection to input size
#         self.adapter_up = nn.Linear(self.down_sample, self.input_size)
#
#         # If we want to have a layer norm on output, we apply it later after a separate residual connection
#         # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
#         if self.add_layer_norm_after:
#             self.adapter_norm_after = nn.LayerNorm(self.input_size)
#
#         # if we want to initialize with the bert strategy then this function is called for all the linear layers
#         if init_bert_weights:
#             self.adapter_down.apply(self.init_bert_weights)
#             self.adapter_up.apply(self.init_bert_weights)
#
#     def forward(self, x, residual_input):  # , residual_input=None):
#         down = self.adapter_down(x)
#
#         up = self.adapter_up(down)
#
#         output = up
#
#         # apply residual connection before layer norm if configured in this way
#         if self.residual_before_ln:
#             output = output + residual_input
#
#         # apply layer norm if available
#         if self.add_layer_norm_after:
#             output = self.adapter_norm_after(output)
#
#         # if residual should be applied after layer norm, apply it here
#         if not self.residual_before_ln:
#             output = output + residual_input
#
#         return output, down, up
#
#     # This is copied from the BertPreTrainedModel class to make this a self containing class.
#     @staticmethod
#     def init_bert_weights(module):
#         """Initialize the weights."""
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # std defaults to 0.02, this might need to be changed
#             module.weight.data.normal_(mean=0.0, std=0.02)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()


# Adapter Flow
class BertFlow(nn.Module):
    """
    Implementation of an AdapterFlow block.
    """

    def __init__(self, config):
        super(BertFlow, self).__init__()

        self.config = config
        self.output_attentions = config.output_attentions

        self.dense_size = int(config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.final_projection = nn.Linear(self.dense_size * 2, int(config.hidden_size))
        self.relu = torch.nn.ReLU()

        if (
            not self.config.adapter_flow["query"]
            and not self.config.adapter_flow["key"]
            and not self.config.adapter_flow["value"]
        ):
            self.dense = nn.Linear(self.dense_size, 1)

        if self.config.adapter_flow["query"]:
            # # a-to-b layer
            # self.query_a2b = nn.Linear(int(config.hidden_size), self.dense_size)
            # self.query_a2b.apply(Adapter.init_bert_weights)
            # # b-to-a layer
            # self.query_b2a = nn.Linear(int(config.hidden_size), self.dense_size)
            # self.query_b2a.apply(Adapter.init_bert_weights)
            # # fusion layer
            self.query_fusion = nn.Linear(int(config.hidden_size), self.dense_size)
            self.query_fusion.apply(Adapter.init_bert_weights)

        if self.config.adapter_flow["key"]:
            # # a-to-b layer
            # self.key_a2b = nn.Linear(int(config.hidden_size), self.dense_size)
            # self.key_a2b.apply(Adapter.init_bert_weights)
            # # b-to-a layer
            # self.key_b2a = nn.Linear(int(config.hidden_size), self.dense_size)
            # self.key_b2a.apply(Adapter.init_bert_weights)
            # fusion layer
            self.key_fusion = nn.Linear(int(config.hidden_size), self.dense_size)
            self.key_fusion.apply(Adapter.init_bert_weights)

        if self.config.adapter_flow["value"]:
            # value matrix for adapter a
            # self.value_a = nn.Linear(int(config.hidden_size), int(config.hidden_size), bias=False)
            # self.value_a.apply(Adapter.init_bert_weights)
            # if self.config.adapter_flow["value_initialized"]:
            #     self.value_a.weight.data = (
            #         torch.zeros(int(config.hidden_size), int(config.hidden_size)) + 0.000001
            #     ).fill_diagonal_(1.0)
            # # value matrix for adapter b
            # self.value_b = nn.Linear(int(config.hidden_size), int(config.hidden_size), bias=False)
            # self.value_b.apply(Adapter.init_bert_weights)
            # if self.config.adapter_flow["value_initialized"]:
            #     self.value_b.weight.data = (
            #             torch.zeros(int(config.hidden_size), int(config.hidden_size)) + 0.000001
            #     ).fill_diagonal_(1.0)
            # value matrix for fusion
            self.value_fusion = nn.Linear(int(config.hidden_size), int(config.hidden_size), bias=False)
            self.value_fusion.apply(Adapter.init_bert_weights)
            if self.config.adapter_flow["value_initialized"]:
                self.value_fusion.weight.data = (
                        torch.zeros(int(config.hidden_size), int(config.hidden_size)) + 0.000001
                ).fill_diagonal_(1.0)

        if self.config.adapter_flow["temperature"]:
            self.T = 50.0
        else:
            self.T = 1.0
        self.reduction = self.T / 1000.0

    def attention(self, query, key, value, residual, type='flow'):
        # Take the dot product between "query" and "key" to get the raw attention scores.
        if type == 'flow':
            # query: [batch_size, seq_len, hidden_dim]
            # key: [batch_size, seq_len, hidden_dim]
            # value: [batch_size, seq_len, hidden_dim]
            # shape: [batch_size, seq_len, seq_len]
            attention_scores = torch.matmul(query, key.transpose(-2, -1))
        else:
            # query: [batch_size, seq_len, hidden_dim]
            # key: [batch_size, seq_len, hidden_dim]
            # value: [batch_size, seq_len, hidden_dim]
            # shape: []
            attention_scores = torch.squeeze(torch.matmul(query.unsqueeze(2), key.transpose(-2, -1)), dim=2)
        # shape: []
        attention_scores = self.dropout(attention_scores)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T)
        self.T = max(self.T - self.reduction, 1.0)
        # if not self.training:
        #     self.recent_attention = attention_probs.detach().cpu().numpy()
        if type == 'flow':
            context_layer = torch.matmul(attention_probs, value)
        else:
            context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value), dim=2)

        # if self.config.adapter_flow["value"] and not self.config.adapter_flow["value_before_softmax"]:
        #     # key/value have dims => batch, toks, number-of-adapters, feats
        #     context_layer = self.value(context_layer)
        # else:
        #     context_layer = context_layer

        # if not self.config.adapter_flow["residual_before"]:
        # context_layer = self.relu(context_layer)

        context_layer += residual
        # if type == 'flow':
        #     return self.relu(context_layer)
        # else:
        #     return context_layer
        return context_layer

    def forward(self, query, key, value, residual):
        # query: shape: [batch_size, seq_len, d_ml] | the hidden states
        # key, value: shape: [num_adapters, batch_size, seq_len, d_ml] | the adapter outputs
        # if self.config.adapter_flow["residual_before"]:
        #     value += residual[None, :, :, :].repeat(value.size(0), 1, 1, 1)

        # 1. Flow modules
        # 1.1 a-to-b flow
        # # shape: [batch_size, seq_len, hidden_dim]
        # query_a2b_out = self.query_a2b(value[1])
        # # shape: [batch_size, seq_len, hidden_dim]
        # key_a2b_out = self.key_a2b(key[0])
        # # shape: [batch_size, seq_len, hidden_dim]
        # value_b_out = self.value_b(value[1])
        # # shape: [batch_size, seq_len, hidden_dim]

        key = self.key_fusion(key)
        value_b_out = self.attention(value[1], key[0], value[0], value[1], 'flow')
        value_a_out = self.attention(value[0], key[1], value[1], value[0], 'flow')
        value = torch.cat((value_a_out.unsqueeze(0), value_b_out.unsqueeze(0)), 0)
        query = self.query_fusion(query)
        value = self.value_fusion(value)
        # value = self.attention(query, key[0], value[1], value[0], 'flow')

        # 1.2 b-to-a flow
        # shape: [batch_size, seq_len, hidden_dim]
        # query_b2a_out = self.query_b2a(value[0])
        # # shape: [batch_size, seq_len, hidden_dim]
        # key_b2a_out = self.key_b2a(key[1])
        # # shape: [batch_size, seq_len, hidden_dim]
        # value_a_out = self.value_a(value[0])
        # # shape: [batch_size, seq_len, hidden_dim]
        # value_a_out = self.attention(query_b2a_out, key_b2a_out, value_a_out, value[0], 'flow')

        # 2. Fusion module

        # shape: [batch_size, seq_len, hidden_dim]
        # value_fusion_out = torch.cat((value[0],
        #                               value[1]),
        #                              2)
        # value_fusion_out = self.final_projection(value_fusion_out)
        value_fusion_out = self.attention(
            query,
            key.permute(1, 2, 0, 3),
            value.permute(1, 2, 0, 3),
            residual,
            "fusion",
        )

        return value_fusion_out


# Adapter Fusion


class BertFusion(nn.Module):
    """
    Implementation of an AdapterFusion block.
    """

    def __init__(self, config):
        super(BertFusion, self).__init__()
        # if config.hidden_size % config.num_attention_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.config = config
        self.output_attentions = config.output_attentions

        self.dense_size = int(config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if (
            not self.config.adapter_fusion["query"]
            and not self.config.adapter_fusion["key"]
            and not self.config.adapter_fusion["value"]
        ):
            self.dense = nn.Linear(self.dense_size, 1)

        if self.config.adapter_fusion["query"]:
            self.query = nn.Linear(int(config.hidden_size), self.dense_size)
            self.query.apply(Adapter.init_bert_weights)

        if self.config.adapter_fusion["key"]:
            self.key = nn.Linear(self.dense_size, self.dense_size)
            self.key.apply(Adapter.init_bert_weights)

        if self.config.adapter_fusion["value"]:
            self.value = nn.Linear(int(config.hidden_size), int(config.hidden_size), bias=False)
            self.value.apply(Adapter.init_bert_weights)
            if self.config.adapter_fusion["value_initialized"]:
                self.value.weight.data = (
                    torch.zeros(int(config.hidden_size), int(config.hidden_size)) + 0.000001
                ).fill_diagonal_(1.0)

        if self.config.adapter_fusion["temperature"]:
            self.T = 50.0
        else:
            self.T = 1.0
        self.reduction = self.T / 1000.0

    def forward(self, query, key, value, residual):

        if self.config.adapter_fusion["residual_before"]:
            value += residual[:, :, None, :].repeat(1, 1, value.size(2), 1)

        if self.config.adapter_fusion["query"]:
            query_layer = self.query(query)
        else:
            query_layer = query

        if self.config.adapter_fusion["key"]:
            key_layer = self.key(key)
        else:
            key_layer = key

        if self.config.adapter_fusion["value"] and self.config.adapter_fusion["value_before_softmax"]:
            # key/value have dims => batch, toks, number-of-adapters, feats
            value_layer = self.value(value)
        else:
            value_layer = value

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.squeeze(torch.matmul(query_layer.unsqueeze(2), key_layer.transpose(-2, -1)), dim=2)

        attention_scores = self.dropout(attention_scores)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T)
        self.T = max(self.T - self.reduction, 1.0)

        if not self.training:
            self.recent_attention = attention_probs.detach().cpu().numpy()

        context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_layer), dim=2)

        if self.config.adapter_fusion["value"] and not self.config.adapter_fusion["value_before_softmax"]:
            # key/value have dims => batch, toks, number-of-adapters, feats
            context_layer = self.value(context_layer)
        else:
            context_layer = context_layer

        if not self.config.adapter_fusion["residual_before"]:
            context_layer += residual

        return context_layer


# Invertible Adapters


def get_subnet_constructor(non_linearity, reduction_factor):
    def subnet(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, dims_in // reduction_factor),
            Activation_Function_Class(non_linearity),
            nn.Linear(dims_in // reduction_factor, dims_out),
        )

    return subnet


class NICECouplingBlock(nn.Module):
    """Coupling Block following the NICE design."""

    def __init__(self, dims_in, dims_c=[], non_linearity="relu", reduction_factor=2):
        super().__init__()

        channels = dims_in[0][0]
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        assert all(
            [dims_c[i][1:] == dims_in[0][1:] for i in range(len(dims_c))]
        ), "Dimensions of input and one or more conditions don't agree."
        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        subnet_constructor = get_subnet_constructor(non_linearity, reduction_factor)
        self.F = subnet_constructor(self.split_len2 + condition_length, self.split_len1)
        self.G = subnet_constructor(self.split_len1 + condition_length, self.split_len2)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[:, :, : self.split_len1], x[:, :, self.split_len1 :])
        if not rev:  # this part after the initial embedding layer
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            y1 = x1 + self.F(x2_c)
            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            y2 = x2 + self.G(y1_c)
        else:  # the part before the last embedding layer
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            y2 = x2 - self.G(x1_c)
            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            y1 = x1 - self.F(y2_c)

        return torch.cat((y1, y2), -1)

    def jacobian(self, x, rev=False):
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class GLOWCouplingBlock(nn.Module):
    """
    Coupling Block following the GLOW design. The only difference to the RealNVP coupling blocks, is the fact that it
    uses a single subnetwork to jointly predict [s_i, t_i], instead of two separate subnetworks. This reduces
    computational cost and speeds up learning. clamp: Soft clamping for the multiplicative component. The amplification
    or attenuation of each input dimension can be at most ±exp(clamp).
    """

    def __init__(self, dims_in, dims_c=[], non_linearity="relu", reduction_factor=2, clamp=5.0):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = math.exp(clamp)
        self.min_s = math.exp(-clamp)

        assert all(
            [tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]
        ), f"Dimensions of input and one or more conditions don't agree: {dims_c} vs {dims_in}."
        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        subnet_constructor = get_subnet_constructor(non_linearity, reduction_factor)
        self.s1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2 * 2)
        self.s2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1 * 2)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[:, :, : self.split_len1], x[:, :, self.split_len1 :])

        if not rev:
            s2, t2 = x1.clone(), x2.clone()
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(torch.cat([y1, *c], 1) if self.conditional else y1)
            s1, t1 = r1[:, : self.split_len2], r1[:, self.split_len2 :]
            y2 = self.e(s1) * x2 + t1
            self.last_jac = torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims + 1))) + torch.sum(
                self.log_e(s2), dim=tuple(range(1, self.ndims + 1))
            )

        else:  # names of x and y are swapped!
            r1 = self.s1(torch.cat([x1, *c], 1) if self.conditional else x1)
            s1, t1 = r1[:, : self.split_len2], r1[:, self.split_len2 :]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(torch.cat([y2, *c], 1) if self.conditional else y2)
            s2, t2 = r2[:, : self.split_len1], r2[:, self.split_len1 :]
            y1 = (x1 - t2) / self.e(s2)
            self.last_jac = -torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims + 1))) - torch.sum(
                self.log_e(s2), dim=tuple(range(1, self.ndims + 1))
            )

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims
