#!/usr/bin/env python3
# -*- coding:gb18030 -*-
import json
import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from nlp_toolbox.modules.losses.label_smoothing_cross_entropy_loss import LabelSmoothingCrossEntropyLoss

def swish(x):
    return x * torch.sigmoid(x)

def gelu(x):
    """ 
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "mish": mish}

class BertConfig(object):
    def __init__(
        self,
        vocab_size=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pool_out_size=None,
        **kwargs,
    ):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # ����������
        self.pool_out_size = hidden_size if pool_out_size is None else pool_out_size

        self.other_config = kwargs
        for attr, value in kwargs.items():
            setattr(self, attr, value)
            logging.info("extra bert config: {} = {}".format(attr, value))


class BertLayerNorm(nn.Module):
    """LayerNorm��, ��Transformer(һ), ��������(encoder)�ĵ�3����"""
    def __init__(self, hidden_size, eps=1e-12, conditional=False):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.conditional = conditional
        if conditional == True:
            #˵�������� ln
            self.weight_dense = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.weight_dense.weight.data.uniform_(0, 0)
            self.bias_dense = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.bias_dense.weight.data.uniform_(0, 0)

    def forward(self, x):
        if self.conditional == False:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.gamma * x + self.beta
        else :
            inputs = x[0]
            cond = x[1]
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)

            weight = self.gamma + self.weight_dense(cond)
            bias = self.beta + self.bias_dense(cond)
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            x = (inputs - u) / torch.sqrt(s + self.variance_epsilon)

            return weight * x + bias



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #logging.info("self.training: {}".format(self.training))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.dropout = lambda x: nn.Dropout(config.hidden_dropout_prob)(x) if self.training else x

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        # input_ids shape: [batch_size, seq_length]

        logging.debug("input_shape: {}".format(input_ids.shape))

        input_shape = input_ids.size()

        seq_length = input_shape[1]
        device = input_ids.device
        # ���û�д�position_ids ��Ĭ��Ϊ����[0,seq_length)
        if position_ids is None:
            # cur shape: [seq_length]
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            # cur shape: [batch_size, seq_length]
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # position_ids shape: [batch_size, seq_length]
        logging.debug("position_ids shape: {}".format(position_ids.shape))

        # ���û�д���token_type_ids Ĭ��Ϊͬһ�仰 ���idȫΪ0
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # inputs_embeds shape : [batch_size, seq_length, hidden_size]
        inputs_embeds = self.word_embeddings(input_ids)
        logging.debug("inputs_embeds shape: {}".format(inputs_embeds.shape))

        # position_embeddings shape : [batch_size, seq_length, hidden_size]
        position_embeddings = self.position_embeddings(position_ids)
        logging.debug("position_embeddings shape: {}".format(position_embeddings.shape))

        # token_type_embeddings shape : [batch_size, seq_length, hidden_size]
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        logging.debug("token_type_embeddings shape: {}".format(token_type_embeddings.shape))

        # ȫ������
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        logging.debug("embeddings shape: {}".format(embeddings.shape))

        # layernorm + dropout
        embeddings = self.LayerNorm(embeddings)
        logging.debug("embeddings shape: {}".format(embeddings.shape))

        embeddings = self.dropout(embeddings)
        logging.debug("cur training mode: {}".format(self.training))
        logging.debug("embeddings shape: {}".format(embeddings.shape))
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        # ��ͷע������ͷ��
        self.num_attention_heads = config.num_attention_heads
        logging.debug("num_attention_heads: {}".format(self.num_attention_heads))
        # ÿͷ��ע����ά�� ȡ��
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        logging.debug("attention_head_size: {}".format(self.attention_head_size))
        # ��ͷע�����ܹ���ά��
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # x shape: [batch_size, seq_length, self.all_head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # x shape: [batch_size, seq_length, self.num_attention_heads, self.attention_head_size]
        x = x.view(*new_x_shape)

        ## ���xshape (batch_size, num_attention_heads, seq_len, head_size)
        # x shape: [batch_size, self.num_attention_heads, seq_length, self.attention_head_size]
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False
    ):
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        # attention_mask shape: [batch_size, 1, seq_length, seq_length]
        # self attention q,k,vΪͬһ��ֵ

        # ����shape��ͬ: [batch_size, seq_length, self.all_head_size]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # ����shape��ͬ: [batch_size, self.num_attention_heads, seq_length, self.attention_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores shape: [batch_size, self.num_attention_heads, seq_length, seq_length]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_mask: ��mask�Ĳ���Ϊ -10e12��δmask�Ĳ���δ0
        # attention_mask��ֱ����raw_attention_score��� Ȼ���softmax����
        # ��mask�Ĳ��ֻἫ��С �൱�ڱ�����
        # attention_scores shape: [batch_size, self.num_attention_heads, seq_length, seq_length]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs shape: [batch_size, self.num_attention_heads, seq_length, seq_length]
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # ע������Ȩ torch.dot()
        # context_layer: [batch_size, self.num_attention_heads, seq_length, self.attention_head_size]
        context_layer = torch.matmul(attention_probs, value_layer)

        # �Ѽ�Ȩ���V reshape, �õ�[batch_size, length, embedding_dimension]
        # context_layer: [batch_size, seq_length, self.num_attention_heads, self.attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer: [batch_size, seq_length, self.all_head_size]
        context_layer = context_layer.view(*new_context_layer_shape)

        # �õ����
        if output_attentions:
            return context_layer, attention_probs
        return context_layer, None


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # hidden_states shape: [batch_size, seq_length, all_head_size(config.hidden_size)]
        # input_tensor shape: [batch_size, seq_length, config.hidden_size]

        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        # projection
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # add + layer norm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False
    ):
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        # attention_mask shape: [batch_size, 1, seq_length, seq_length]

        # multi head attention
        # self_outputs: [batch_size, seq_length, all_head_size]�����У�all_head_size==config.hidden_size
        # attention_matrix shape: [batch_size, self.num_attention_heads, seq_length, seq_length]
        self_outputs, attention_metrix = self.self(hidden_states, attention_mask, output_attentions=output_attentions)

        # add + layer norm
        # attention_output shape: [batch_size, seq_length, config.hidden_size]
        attention_output = self.output(self_outputs, hidden_states)

        return attention_output, attention_metrix


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] ## relu 

    def forward(self, hidden_states):
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        # fc
        # hidden_states shape: [batch_size, seq_length, config.intermediate_size]
        hidden_states = self.dense(hidden_states)
        # �����
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # hidden_states shape: [batch_size, seq_length, config.intermediate_size]
        # input_tensor shape: [batch_size, seq_length, config.hidden_size]

        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # add + layer norm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False
    ):
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        # attention_mask shape: [batch_size, 1, seq_length, seq_length]

        # multi head attention + add + layer norm
        # attention_output shape: [batch_size, seq_length, config.hidden_size]
        # attention_matrix shape: [batch_size, self.num_attention_heads, seq_length, seq_length]
        attention_output, attention_matrix = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)

        # feed forward
        # intermediate_output shape: [batch_size, seq_length, config.intermediate_size]
        intermediate_output = self.intermediate(attention_output)

        # add + layer norm
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, attention_matrix


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=True,
        output_attentions=False
    ):
        all_encoder_layers = []
        all_attention_matrices = []
        for i, layer_module in enumerate(self.layer):

            layer_output, attention_matrix = layer_module(
                hidden_states, attention_mask, output_attentions=output_attentions
            )

            hidden_states = layer_output
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_attention_matrices.append(attention_matrix)

        # ���ֻ������һ�� ������������һ��
        # ���������в� ��ѭ���������� ���ﲻ��Ҫ
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_attention_matrices.append(attention_matrix)

        return all_encoder_layers, all_attention_matrices


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.pool_out_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # avg_pool: True�����һ��avg_pool��Ϊ���������CLSΪ���
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        logging.debug("hidden_states shape: {}".format(hidden_states.shape))
        # hidden_states shape: [batch_size, config.pool_out_size]
        hidden_states = hidden_states[:, 0]

        logging.debug("hidden_states shape: {}".format(hidden_states.shape))
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.decoder.weight = bert_model_embedding_weights

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        logging.debug("hidden_states shape: {}".format(hidden_states.shape))

        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        hidden_states = self.transform(hidden_states)
        logging.debug("hidden_states shape: {}".format(hidden_states.shape))

        # hidden_states shape: [batch_size, seq_length, config.vocab_size]
        hidden_states = self.decoder(hidden_states)
        logging.debug("hidden_states shape: {}".format(hidden_states.shape))
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_path, keep_tokens=None, **kwargs):
        assert "vocab_size" in kwargs, \
                "parameter 'vocab_size' is required when load bert"

        bert_state_dict_path = os.path.join(pretrained_model_path,
                "pytorch_model.bin")
        assert os.path.exists(bert_state_dict_path), \
                "cannot find state dict file: {}".format(bert_state_dict_path)

        bert_config_path = os.path.join(pretrained_model_path, 'config.json')
        assert os.path.exists(bert_config_path), \
                "cannot find bert config file: {}".format(bert_config_path)

        with open(bert_config_path) as rf:
            bert_config_dict = dict(json.loads(rf.read()), **kwargs)
            logging.info("bert_config_dict: {}".format(bert_config_dict))
        bert_config_dict = BertConfig(**bert_config_dict)
        model = cls(bert_config_dict)

        pretrained_state_dict = torch.load(bert_state_dict_path)

        logging.debug("remove state dict: {}".format(
            [k for k in pretrained_state_dict.keys() \
                    if k[:4] != "bert" or "pooler" in k]))

        # ȥ��pooler�ͷ�bert��ص�Ȩ��
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() \
                if k[:4] == "bert" and "pooler" not in k}

        # Ԥѵ�������е�Ȩֵ
        pretrained_state_dict_name_set = pretrained_state_dict.keys()
        # ģ����Ҫ��Ȩ��
        model_state_dict_name_set = model.state_dict().keys()

        logging.info("unused weight in pretrained file: {}".format(
            pretrained_state_dict_name_set - model_state_dict_name_set))
        logging.info("missing weight in pretrained file: {}".format(
            model_state_dict_name_set - pretrained_state_dict_name_set))

        if keep_tokens is not None:
            ## ˵������ʱ��ˣ�embeedding��ҲҪ������
            embedding_weight_name = "bert.embeddings.word_embeddings.weight"

            pretrained_state_dict[embedding_weight_name] = \
                    pretrained_state_dict[embedding_weight_name][keep_tokens]

        model.load_state_dict(pretrained_state_dict, strict=False)
        torch.cuda.empty_cache()
        logging.info("succeed loading model from {}".format(pretrained_model_path))

        return model

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear)):
            # ��ʼ����ӳ���Ĳ���Ϊ��̬�ֲ�
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            # ��ʼ��LayerNorm�е�alphaΪȫ1, betaΪȫ0
            module.beta.data.zero_()
            module.gamma.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # ��ʼ��ƫ��Ϊ0
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)

        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_all_encoded_layers=True,
        output_attentions=False
    ):
        # 0Ϊpad id �����ǽ�pad mask��
        # input ids shape: [batch_size, seq_length]
        logging.debug("input ids shape: {}".format(input_ids.shape))
        logging.debug("input ids[0]: {}".format(input_ids[0]))
        extended_attention_mask = (input_ids > 0).float()


        # ע��������mask: [batch_size, 1, 1, seq_length]
        # extended_attention_mask shape: [batch_size, 1, 1, seq_length]
        extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)
        logging.debug("extended_attention_mask shape1: {}".format(extended_attention_mask.shape))

        if attention_mask is not None :
            ## �����������ע����mask����None���Ǿ�ֱ���ô�������ע����mask �� ԭʼmask
            # ע�� ԭʼmask��extended_attention_mask�������������pad������Ϊ0��ȥ��pad����Ӱ��
            # attention_mask shape: [batch_size, 1, seq_length, seq_length]
            extended_attention_mask = attention_mask * extended_attention_mask
            logging.debug("attention_mask shape: {}".format(attention_mask.shape))

        # extended_attention_mask shape: [batch_size, 1, seq_length, seq_length]
        logging.debug("extended_attention_mask shape2: {}".format(extended_attention_mask.shape))
        logging.debug("extended_attention_mask[0][0]: {}".format(extended_attention_mask[0][0]))

        # ���token_type_idsΪNone��Ĭ��Ϊ��һ�仰
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # token_type_ids shape: [batch_size, seq_length]
        logging.debug("token_type_ids shape: {}".format(token_type_ids.shape))

        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # ��mask�Ĳ���Ϊ -10e12��δmask�Ĳ���δ0
        # extended_attention_mask��ֱ����raw_attention_score��� Ȼ���softmax����
        # ��mask�Ĳ��ֻἫ��С �൱�ڱ�����
        extended_attention_mask = (1.0 - extended_attention_mask) * -10e12
        # extended_attention_mask shape: [batch_size, 1, seq_length, seq_length]
        logging.debug("extended_attention_mask shape3: {}".format(extended_attention_mask.shape))

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        # embedding_output shape: [batch_size, seq_length, config.hidden_size]
        logging.debug("embedding_output shape: {}".format(embedding_output.shape))
        logging.debug("embedding_output[0][0][:20]: {}".format(embedding_output[0][0][:20]))

        # ����config.num_hidden_layers������
        encoder_layers, all_attention_matrices = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_attentions=output_attentions
        )
        logging.debug("encoder_layers size: {}".format(len(encoder_layers)))
        logging.debug("all_attention_matrices size: {}".format(len(all_attention_matrices)))

        # sequence_output shape: [batch_size, seq_length, config.hidden_size]
        sequence_output = encoder_layers[-1]
        logging.debug("sequence_output shape: {}".format(sequence_output.shape))

        # pooled_output shape: [batch_size, config.hidden_size]
        pooled_output = self.pooler(sequence_output)
        logging.debug("pooled_output shape: {}".format(pooled_output.shape))

        if not output_all_encoded_layers:
            # ��������������encoder��
            encoder_layers = encoder_layers[-1]

        if output_attentions:
            return encoder_layers, pooled_output, all_attention_matrices

        # ���ظ������� ������ػ����
        return encoder_layers, pooled_output


class BertForClassification(BertPreTrainedModel):
    """
    """
    def __init__(self, config):
        super(BertForClassification, self).__init__(config)
        self.bert = BertModel(self.config)
        self.final_fc = torch.nn.Linear(
                self.config.hidden_size,
                self.config.num_class)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.label_smooth_ratio = self.config.other_config.get("label_smooth", None)
        logging.info("label_smooth_ratio: {}".format(self.label_smooth_ratio))
        #self.loss_layer = nn.CrossEntropyLoss(reduction="mean") if \
        #        self.label_smooth_ratio is None else LabelSmoothingCrossEntropyLoss(self.label_smooth_ratio)

    def compute_loss(self, softmax_pred, labels):
        """
        ����loss
        softmax_pred: (batch_size, 1)
        """
        softmax_pred = softmax_pred.view(-1, self.config.num_class)
        labels = labels.view(-1)
        if self.label_smooth_ratio is not None:
            loss = LabelSmoothingCrossEntropyLoss(self.label_smooth_ratio)(softmax_pred, labels)
        else:
            # CrossEntropyLoss = Softmax + log + NLLLoss
            loss = torch.nn.NLLLoss(reduction="mean")(torch.log(softmax_pred), labels)

        return loss

    def forward(self, input_ids, token_type_ids=None, position_ids=None, labels=None, use_layer_num=-1, with_softmax=True, **kwargs):
        # Ĭ��-1 ȡ���һ��
        if use_layer_num != -1 and (use_layer_num < 0 or use_layer_num > 11):
            # Խ��
            raise Exception("����ѡ��Χ[0, 12)��Ĭ��Ϊ-1�������һ��")
        enc_layers, _ = self.bert(input_ids, token_type_ids=token_type_ids,
            position_ids=position_ids, output_all_encoded_layers=True)

        squence_out = enc_layers[use_layer_num]

        cls_token = squence_out[:, 0]# ȡ��cls���� ���з���

        predictions = self.final_fc(cls_token)

        softmax_pred = self.softmax(predictions)

        res_dict = {
                "sent_logits": predictions,
                "sent_softmax": softmax_pred,
                }

        if labels is not None:
            ## ����loss
            loss = self.compute_loss(softmax_pred, labels)
            res_dict["loss"] = loss

        return res_dict


class BertForSeqSim(BertPreTrainedModel):
    """����seq2seq ͬʱҲ����sim
    """
    def __init__(self, config):
        self.seq_task = config.other_config.pop("seq_task", False)
        self.sim_task = config.other_config.pop("sim_task", False)
        self.sim_loss_type = config.other_config.pop("sim_loss_type", None)

        super(BertForSeqSim, self).__init__(config)
        self.bert = BertModel(self.config)

        # ��attention_weight����������������������
        # ����Ϊ��������
        # ������BISON : BM25-weighted Self-Attention Framework for Multi-Fields Document Search��github����������
        # https://github.com/cadobe/bison/issues/3
        # �о��������̫��
        #self.is_vertical = config.is_vertical

        # �����seq2seq���� ��Ҫ��decoder
        if self.seq_task:
            self.decoder = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
            self.vocab_size = config.vocab_size
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")

        if self.sim_loss_type is not None:
            if self.sim_loss_type == "both" or self.sim_loss_type == "pairwise":
                self.triplet_margin = config.triplet_margin
                logging.info("triplet_margin: {}".format(self.triplet_margin))
                self.relu = torch.nn.ReLU()
                # ͨ�õ�batch pair wiseʱ����loss������������λ��
                # self.batch_pair_wise_mask[i][j][k]Ϊ1�����������ˣ���Ҫ������������������
                # 1. i,j,k������ͬ��
                #     a. i=j����i=k��ʱ��ij��ik����ʾ��������������ƶȣ�һ��Ϊһ������ѵ��û������
                #     b. j=kʱ��ij��ik��ͬһ��������anchor�����ƶȲ�࣬һ����ȣ�����ѵ��Ҳû������
                # 2. j < k��triplet_loss[i][j][k]��triplet_loss[i][j][k]��һ���ģ�û�б�Ҫ������
                self.batch_pair_wise_mask = None

            if self.sim_loss_type == "both" or self.sim_loss_type == "pointwise":
                self.pos_neg_weight = config.pos_neg_weight
                logging.info("pos_neg_weight: {}".format(self.pos_neg_weight))
        else:
            assert not self.sim_task, "sim_loss_type is required when sim_task on"

    def gen_seq_mask(self, token_type_ids):
        # ����bert seq2seq�����mask

        # ��text_a����Ϊlen_a��text_b����Ϊlen_b
        # ��token_type_id���� = [0, 0, ..., 0, 1, 1, ...,1]
        # ����0��len_a����1��len_b��

        # token_type_id shape: [batch_size, seq_length]
        seq_len = token_type_ids.shape[1]
        ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=token_type_ids.device)
        # a_mask shape: [1, 1, seq_length, seq_length]
        a_mask = ones.tril() # �����Ǿ���

        # ��text_a="�����ɶ", text_b="��ն��"
        # ��text_a����Ϊ6(��cls��sep) text_b����Ϊ4(��sep)
        # seq2seq��attention_maskΪ
        # [CLS] : [ 1 1 1 1 1 1 0 0 0 0]
        # ��    : [ 1 1 1 1 1 1 0 0 0 0]
        # ��    : [ 1 1 1 1 1 1 0 0 0 0]
        # ��    : [ 1 1 1 1 1 1 0 0 0 0]
        # ɶ    : [ 1 1 1 1 1 1 0 0 0 0]
        # [SEP] : [ 1 1 1 1 1 1 0 0 0 0]
        # ��    : [ 1 1 1 1 1 1 1 0 0 0]
        # ��    : [ 1 1 1 1 1 1 1 1 0 0]
        # ��    : [ 1 1 1 1 1 1 1 1 1 0]
        # [SEP] : [ 1 1 1 1 1 1 1 1 1 1]
        # ʵ��Ԥ����,text_b�����һλsep��Ԥ��������,��Ϊ��[SEP]�����ɽ�ֹ

        # s_ex12 shape: [batch_size, 1, 1, seq_length]
        # 1 - ��mask
        s_ex12 = token_type_ids.unsqueeze(1).unsqueeze(2).float()

        # 1 - ��mask
        # s_ex12 shape: [batch_size, 1, seq_length, 1]
        s_ex13 = token_type_ids.unsqueeze(1).unsqueeze(3).float()

        # (1.0 - s_ex12) * (1.0 - s_ex13) shape: [batch_size, 1, seq_length, seq_length]
        # ��maxk*��mask��attentionʱʼ����Ҫ��Ԫ��
        # �������ά[seq_length, seq_length]�ľ�����
        # ���Ϸ�[len_a, len_a]��С�ľ���Ԫ��Ϊ1������Ԫ�ؾ�Ϊ0

        # s_ex13 * a_mask shape: [batch_size, 1, seq_length, seq_length]
        # �����Ǿ���*(1-��mask)��Ҫ��Ϊ��ѵ��ʱ���ܵõ��������Ϣ
        # �������ά[seq_length, seq_length]�ľ�����
        # ǰlen_a��ȫΪ0����len_b�У�ǰlen_aȫΪ1�����½�[len_b, len_b]�ľ���Ϊ�����Ǿ���
        # a_mask shape: [batch_size, 1, seq_length, seq_length]
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask
        return a_mask

    def forward(self,
            input_ids,
            token_type_ids,
            position_ids=None,
            labels=None,
            output_all_encoded_layers=True,
            output_attentions=False,
            ):

        # input_tensor shape: [batch_size, seq_length]
        logging.debug("input_ids shape: {}".format(input_ids.shape))
        #logging.debug("input_tensor[0]: {}".format(input_tensor[0]))
        #logging.debug("input_tensor text: {}".format("/ ".join([self.ix2word[x] for x in input_tensor[0].cpu().numpy()])))

        # token_type_id ����a��pad�Ĳ��ֶ�Ϊ0
        # token_type_id shape: [batch_size, seq_length]
        logging.debug("token_type_ids shape: {}".format(token_type_ids.shape))
        #logging.debug("token_type_ids[0]: {}".format(token_type_ids[0]))
        # position_ids shape: [batch_size, seq_length]
        if position_ids is not None:
            logging.debug("position_ids shape: {}".format(position_ids.shape))

        if labels is not None:
            logging.debug("labels shape: {}".format(labels.shape))
            logging.debug("labels[:20] : {}".format(labels[:20]))

        # ���Ҫ����seq2seq���� ����Ҫ����seq2seq��attention_mask ����ΪNone
        # attention_mask shape: [batch_size, 1, seq_length, seq_length]
        attention_mask = self.gen_seq_mask(token_type_ids) if self.seq_task else None

        # ����bert �õ�������� �� pool_out���
        # ��������shape��[batch_size, seq_length, config.hidden_size]
        enc_layers, pool_out = self.bert(
                input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                output_all_encoded_layers=output_all_encoded_layers,
                output_attentions=output_attentions)

        res_dict = {
                "text_output": pool_out,
                }

        # ȡ���һ��
        # Ԥ���token����������һ��token��Ҳ���ǽ�β��[SEP]��
        # Ԥ������Ҫȥ�����һ��Ԥ���ֵ �ڹ���ѵ�����ݵ�ʱ�� labelҲ��������seq_length��1
        # enc_layers[-1] shape: [batch_size, seq_length, config.hidden_size]
        # predictions shape: [batch_size, seq_length-1, self.vocab_size] or None
        if self.seq_task:
            predictions = self.decoder(enc_layers[-1])[:, :-1].contiguous()
            res_dict["token_output"] = predictions

        if labels is not None:
            # ����loss
            total_loss = 0.0
            if self.seq_task:
                # ����lossʱֻ����target_mask��Ϊ1�� ����ȥ����һ��Ϊ0�� ��predictions��labels��ά�ȶ���
                # ����ȡǰ�� ����Ϊ�ڹ���labelʱ Ҳ��ȥ����ǰһ�� �Դﵽԭ��������һλ
                # ��ʱlabels���Ǹ�λ�õ���һ��λ����Ҫ�����vocab���б�
                # ��������ҲӦ�ô�ǰ��ȥ��
                # Ч���� target_mask�����Ǵӵ�һ��SEP��������SEP��֮��Ϊ1 ��ʾ�ڶ��仰��ʼ
                # ����ǰ��ȥ��һ��
                # ��target_mask�ӵ�һ��SEP������SEP����ʼΪ1����ʾ�ӵ�һ��SEP��ʼ��ҪԤ����һ��vocab��ʲô
                target_mask = token_type_ids[:, 1:].contiguous()
                logging.debug("target_mask[0]: {}".format(target_mask[0]))
                # �������������һλ ���������Ŀ��ֵ
                seq_labels = input_ids[:, 1:].contiguous()
                total_loss += self.compute_seq_loss(predictions, seq_labels, target_mask)

            if self.sim_task:
                total_loss += self.compute_sim_loss(pool_out, labels)

            logging.debug("total_loss: {}".format(total_loss))

            res_dict["loss"] = total_loss

        return res_dict

    def compute_seq_loss(self, predictions, labels, target_mask):
        # ��ƽ
        # predictions shape: [batch_size*(seq_length-1), self.vocab_size]
        predictions = predictions.view(-1, self.vocab_size)

        # ��ƽ
        # labels shape: [batch_size*(seq_length-1)]
        labels = labels.view(-1)
        logging.debug("labels shape: {}".format(labels.shape))

        # ֻ��text_b ��Ϊ1����ҪԤ��
        target_mask = target_mask.view(-1).float()
        # ����loss
        # ͨ��mask ȡ�� pad �;���a����Ԥ���Ӱ��
        loss = (self.ce_loss(predictions, labels) * target_mask).sum() / target_mask.sum()
        logging.debug("seq loss: {}".format(loss))
        return loss

    def compute_sim_loss(self, pool_out, labels):
        """
        target_mask : ����a���ֺ�pad����ȫΪ0�� ������b����Ϊ1
        """
        logging.debug("pool_out shape: {}".format(pool_out.shape))
        logging.debug("labels shape: {}".format(labels.shape))
        # �������ƶȵ�loss
        sim_loss = {
                "pointwise": self.compute_point_wise_loss,
                "pairwise": self.compute_pair_wise_loss,
                "both": self.compute_both_loss,
                }[self.sim_loss_type](pool_out, labels)
        logging.debug("sim_loss = {}".format(sim_loss))
        return sim_loss

    def compute_both_loss(self, text_vecs, labels):
        return self.compute_point_wise_loss(text_vecs, labels) + \
                self.compute_pair_wise_loss(text_vecs, labels)

    def compute_pair_wise_loss(self, text_vecs, labels):
        # �������ƶȵ�Ŀ����� label_idһ�µ�������
        batch_size = labels.shape[0]
        logging.debug("batch size: {}".format(batch_size))
        labels = labels.unsqueeze(dim=0)
        logging.debug("labels shape: {}".format(labels.shape))
        logging.debug("labels: {}".format(labels))
        # Ŀ��ƥ�����
        # target_label shape: [batch_size, batch_size]
        target_label = (labels.t() == labels) * 1
        logging.debug("target_label shape: {}".format(target_label.shape))
        logging.debug("target_label: {}".format(target_label))

        # �����batch�����ƶ�
        # ��һ��������
        text_vecs = nn.functional.normalize(text_vecs, dim=1)
        logging.debug("text_vecs: {}".format(text_vecs))
        # �������ƶ�
        cos_sim_vec = torch.mm(text_vecs, text_vecs.t())
        logging.debug("cos_sim_vec shape: {}".format(cos_sim_vec.shape))
        logging.debug("cos_sim_vec: {}".format(cos_sim_vec))

        #batch_size = 4
        #target_label = torch.tensor([
        #    [1, 1, 0, 1],
        #    [0, 1, 1, 0],
        #    [0, 0, 1, 1],
        #    [1, 0, 1, 0],
        #], device=labels.device)

        #cos_sim_vec = torch.tensor([
        #    [1.0, 0.8, 0.4, 0.9],
        #    [0.5, 1.0, 0.4, 0.2],
        #    [0.2, 0.3, 1.0, 0.5],
        #    [0.4, 0.3, 0.3, 1.0],
        #], device=labels.device)

        # ����triplet loss����triplet_loss[i][j][k] = triplet_loss(cos_sim_ij, cos_sim_ik)
        # triplet_loss(cos_sim_pos, cos_sim_neg, margin) = cos_sim_neg - cos_sim_pos + margin

        # �ȼ���
        # cos_sim_minus[i][j][k] = cos_sim_vec[i][j] - cos_sim_vec[i][k]
        # ����ij�����ƶ�������ik�����ƶȵĲ��
        # cos_sim_minus shape: [batch_size, batch_size, batch_size]
        cos_sim_minus = cos_sim_vec.unsqueeze(2) - cos_sim_vec.unsqueeze(1)

        # ��Ϊ֮��ֻ�ῼ��ij��jk һ����������һ���Ǹ����������
        # ��������ֻ��ij�Ƿ�Ϊ������
        # ��Ϊtriplet_loss��Ҫ���������ƶ�-���������ƶ�
        # ������ijΪ������ʱ triplet_loss = cos_sim_ij - cos_sim_ik + margin
        # ������ijΪ������ʱ triplet_loss = cos_sim_ik - cos_sim_ij + margin
        # ����triplet_loss_ijk = pos_flag_ij * cos_sim_minus_ijk + margin
        # ����ijΪ������ʱ pos_flag_ij = 1 ���� pos_flag_ij = -1
        # pos_flag_ij shape: [batch_size, batch_size, batch_size]
        pos_flag_ij = (1 - 2 * target_label).unsqueeze(2).repeat(1, 1, batch_size)

        # �õ�����ijk��triplet loss
        # triplet_loss shape: [batch_size, batch_size, batch_size]
        triplet_loss =  self.relu(cos_sim_minus * pos_flag_ij + self.triplet_margin)

        # �õ�Ŀ��triplet_loss�������

        # ��������ͬbatch_size��ͨ�õ�����
        if self.batch_pair_wise_mask is None or self.batch_pair_wise_mask.shape[0] != batch_size:
            # forѭ���߼���� ��̫�� batch_size=128ʱ ��ʱ15������
            # ���������߼����� ���� batch_size=128ʱ ��ʱ0.087��
            ## ====================== forѭ�� չʾ�߼� =========================
            #logging.info("creat batch_pair_wise_mask")
            #self.batch_pair_wise_mask = torch.zeros((batch_size, batch_size, batch_size), dtype=torch.int)
            #for anchor_ind in range(batch_size):
            #    for i_ind in range(batch_size):
            #        # triplet��������Ҫ������ͬ
            #        if i_ind == anchor_ind:
            #            continue
            #        for j_ind in range(i_ind + 1, batch_size):
            #            # triplet��������Ҫ������ͬ
            #            if j_ind == anchor_ind:
            #                continue
            #            # self.batch_pair_wise_mask[i][j][k]Ϊ1�������ڵ�����Ҫ������������������
            #            # 1. i,j,k������ͬ��
            #            #     a. i=j����i=k��ʱ��ij��ik����ʾ��������������ƶȣ�һ��Ϊһ������ѵ��û������
            #            #     b. j=kʱ��ij��ik��ͬһ��������anchor�����ƶȲ�࣬һ����ȣ�����ѵ��Ҳû������
            #            # 2. j < k��triplet_loss[i][j][k]��triplet_loss[i][j][k]��һ���ģ�û�б�Ҫ������
            #            self.batch_pair_wise_mask[anchor_ind][i_ind][j_ind] = 1
            #self.batch_pair_wise_mask = self.batch_pair_wise_mask.to(labels.device)
            ## ========================== ������� =============================
            # ������������������� ���������forѭ��һ��
            # ����һ��shapeΪ[batch_size, batch_size]��������ȫһ���� �������Խ���
            # �������һά�ȣ�����batch_size��
            # ��shapeΪ[batch_size, batch_size, batch_size]�Ļ�������k
            kernel = torch.triu(torch.ones((batch_size, batch_size), device=labels.device), diagonal=1)\
                    .unsqueeze(0).repeat(batch_size, 1, 1)
            logging.debug("kernel: {}".format(kernel))

            # �ڻ�������������0 ����kernel[i]�������i�к͵�i�ж�Ӧ��Ϊ0

            # Ϊ������һ��[batch_size, batch_size]�ĶԽǾ��� �����һά�� ����batch_size��
            # ��shapeΪ[batch_size, batch_size, batch_size]�Ĺ��˾���f
            # Ȼ�����i����[0, batch_size-1] ��f[i][i][i]=0 ���ڶ�ʵ�ֶ�kernel[i]�������i�к͵�i�ж���0
            filter_matrix = torch.eye(batch_size, device=labels.device).unsqueeze(0).repeat(batch_size, 1, 1)
            for ind in range(batch_size):
                filter_matrix[ind][ind][ind] = 0
            logging.debug("filter_matrix: {}".format(filter_matrix))

            # matmul(f,k)���Խ�k�ĵ�i������ĵ�i��ȫ��0
            # matmul(k,f)���Խ�k�ĵ�i������ĵ�i��ȫ��0
            # ���յõ�Ŀ�����
            self.batch_pair_wise_mask = torch.matmul(torch.matmul(filter_matrix, kernel), filter_matrix)
            logging.debug("batch_pair_wise_mask: {}".format(self.batch_pair_wise_mask))

        # ֻ����һ��һ��������loss
        # ͨ�õ�pairwise_mask * ������ǩ�෴��pair
        target_mask = self.batch_pair_wise_mask * (target_label.unsqueeze(2) != target_label.unsqueeze(1))
        logging.debug("target_mask shape: {}".format(target_mask.shape))
        logging.debug("target_mask: {}".format(target_mask))

        # ֻ����Ŀ��triplet_loss
        triplet_loss *= target_mask
        logging.debug("triplet_loss: {}".format(triplet_loss))

        # ͳ�ƴ���triplet_loss����Ŀ
        triplet_loss_active_num = (triplet_loss > 0).sum()
        logging.debug("triplet_loss_active_num: {}".format(triplet_loss_active_num))
        logging.debug("target_mask: {}".format(target_mask.sum()))

        # ͳ��pairwise��loss
        mean_loss = triplet_loss.sum() / target_mask.sum() #triplet_loss_active_num
        logging.debug("triplet_loss sum: {}".format(triplet_loss.sum()))
        logging.debug("target sum: {}".format(target_mask.sum()))
        logging.debug("mean_loss: {}".format(mean_loss))

        return mean_loss

    def compute_point_wise_loss(self, cls_vecs, labels):
        logging.debug("cls_vecs: {}".format(cls_vecs))
        logging.debug("labels: {}".format(labels))
        # �������ƶȵ�Ŀ����� label_idһ�µ�������
        batch_size = labels.shape[0]
        logging.debug("batch size: {}".format(batch_size))
        labels = labels.unsqueeze(dim=0)
        logging.debug("labels shape: {}".format(labels.shape))
        logging.debug("labels: {}".format(labels))
        # Ŀ��ƥ�����
        # target_label shape: [batch_size, batch_size]
        target_label = (labels.t() == labels) * 1
        logging.debug("target_label shape: {}".format(target_label.shape))
        logging.debug("target_label: {}".format(target_label))

        # ����loss�Ǻ��ԶԽ���
        # ƥ��Ŀ���������
        target_mask = 1 - torch.eye(target_label.shape[0], device=labels.device)

        if self.pos_neg_weight:
            # ������������
            pos_label_num = torch.sum(target_label)
            logging.debug("pos_label_num: {}".format(pos_label_num))
            neg_label_num = batch_size ** 2 - batch_size - pos_label_num
            logging.debug("neg_label_num: {}".format(neg_label_num))
            neg_pos_ratio = torch.true_divide(neg_label_num, pos_label_num)
            logging.debug("neg_pos_ratio: {}".format(neg_pos_ratio))
            # �������������������������ʧȨ��
            # ����ƥ��������Ȩ��
            # ��ƥ�䴦Ȩ��Ϊ1 ƥ�䴦Ȩ��=neg��/pos��
            # �Դ�ƽ������������
            # ƽ�����������ɼ���ģ��ѵ��
            target_weight = (target_label * neg_pos_ratio) + (1 - target_label)

            target_weight *= target_mask
        else:
            # ��������ʧȨ��һ��
            target_weight = target_mask

        # �����batch�����ƶ�
        # ��һ��������
        cls_vecs = nn.functional.normalize(cls_vecs, dim=1)
        # �������ƶ�
        cls_vecs = torch.mm(cls_vecs, cls_vecs.t())
        logging.debug("cls_vecs shape: {}".format(cls_vecs.shape))
        logging.debug("cls_vecs: {}".format(cls_vecs))


        # ����pointwise��loss
        # ������Ŀ�����Ĳ��
        # TODO hard negative ÿ������ֻ�������ѵ���һ����loss
        dis_vecs = (target_label - cls_vecs).pow(2)

        # ��Ϊsqrt(x)��ʱ��x�Ķ�����Ϊ(0, +inf) ��xΪ0ʱ sqrt�޵���
        # �Ӹ���Сֵ ��ֹsqrt�Ĳ���ֵΪ0 ���·��򴫲�ʱ����
        sim_loss = torch.sqrt((dis_vecs * target_weight).sum() / target_mask.sum() + 1e-8)
        logging.debug("pointwise sim_loss: {}".format(sim_loss))
        return sim_loss
