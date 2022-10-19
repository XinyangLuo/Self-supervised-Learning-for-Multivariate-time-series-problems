import math
import torch
from torch import nn
from utils import *

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, 
                                            num_hiddens, num_heads, dropout,
                                            use_bias, **kwargs)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(
            X, self.attention(X, X, X, valid_lens)
        )
        return self.addnorm2(Y, self.ffn(Y))


class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)


    def forward(self, X, state):
        # State contains: encoder ouput, encoder length, previous decoder output
        enc_outputs, enc_valid_lens = state[0], state[1]
        # If none, use X as key and values
        if state[2][self.i] is None:
            key_values = X
        # Else, concat previous output
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if self.training:
            # when training create a mask with shape (batch size, sequence length)
            batch_size, num_steps, _ = X.shape
            # for each word in the sequence, valid length equal to its position
            # in this sequence. Each row of mask is [1, 2, 3, ..., num_steps].
            # Then prediciton of the next word will be only made by the previous
            # words. While training, The decoder takes a sentence as input and 
            # output a whole sentence, not word by word.

            # So, in time series, if we want to predict x_{t+1} to x_{t+11} we
            # input x_t to x_{t+10} to the decoder
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            # when prediction, there is no need for valid length
            dec_valid_lens = None
        

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerEncoder(nn.Module):
    def __init__(self, seq_length, feature_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.input = nn.Linear(feature_size, num_hiddens)
        #self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.pos_encoding = LearnablePosEncoding(num_hiddens, seq_length)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('block {}'.format(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens, 
                norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, 
                use_bias))

    def forward(self, X, valid_lens, *args):
        # sqrt(self.num_hiddens) is to ensure feature has the similar abosulute 
        # value to positional encoding
        #X = self.pos_encoding(self.input(X)*math.sqrt(self.num_hiddens))
        X = self.pos_encoding(self.input(X))
        self.attention_weights = [None]*len(self.blks)

        # save the log of attention weights
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights

        return X


class TransformerDecoder(nn.Module):
    def __init__(self, seq_length, feature_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.input = nn.Linear(feature_size, num_hiddens)
        #self.pos_encoding = PositionalEncoding(num_hiddens, dropout, seq_length)
        self.pos_encoding = LearnablePosEncoding(num_hiddens, seq_length)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block {}".format(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))

        # output layer
        self.dense = nn.Linear(num_hiddens, feature_size)


    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None]*self.num_layers]


    def forward(self, X, state):
        #X = self.pos_encoding(self.input(X)*math.sqrt(self.num_hiddens))
        X = self.pos_encoding(self.input(X))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights

        return self.dense(X), state


    def attention_weights(self):
        return self._attention_weights


class TimeTransformer(nn.Module):
    def __init__(self, seq_length, feature_size, key_size, query_size, 
                 value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout, mask_prob=0.3,
                 attn_bias = False, **kwargs):
        super(TimeTransformer, self).__init__(**kwargs)
        self.encoder = TransformerEncoder(seq_length, feature_size, key_size, 
                query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                ffn_num_hiddens, num_heads, num_layers, dropout, attn_bias, **kwargs)
        self.decoder = TransformerDecoder(seq_length, feature_size, key_size, 
                query_size, value_size, num_hiddens, norm_shape, ffn_num_input, 
                ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs)
        self.mask_prob = mask_prob

    
    def forward(self, enc_X, dec_X):
        if self.training:
            self.mask = (torch.rand(enc_X.shape, device=enc_X.device) < self.mask_prob).detach()
            enc = self.encoder(enc_X*(~self.mask), None)
            state = self.decoder.init_state(enc, None)
            out_X, out_state = self.decoder(dec_X*(~self.mask), state)
        return out_X


class TimeTransformerClassifier(nn.Module):
    def __init__(self, seq_length, feature_size, key_size, query_size, 
                value_size, num_hiddens, norm_shape, ffn_num_input,
                ffn_num_hiddens, num_heads, num_layers, dropout, num_cat,
                attn_bias=False, **kwargs):
        super(TimeTransformerClassifier, self).__init__(**kwargs)
        self.encoder = TransformerEncoder(seq_length, feature_size, key_size,
            query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
            ffn_num_hiddens, num_heads, num_layers, dropout, attn_bias)
        self.conv = nn.Conv1d(seq_length, 1, 1)
        self.out = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_cat)
        )


    def forward(self, x):
        enc_x = self.encoder(x, None)
        conv_x = self.conv(enc_x)
        return self.out(conv_x)


class DualNetEncoder(nn.Module):
    def __init__(self, seq_length, feature_size, key_size, query_size, 
                 value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout, kernel_size,
                 attn_bias=False, conv_bias=False, **kwargs):
        super(DualNetEncoder, self).__init__(**kwargs)
        self.conv = CasualConv1d(feature_size, feature_size, kernel_size,
                                 bias=conv_bias)
        self.maxpool = CasualMaxPool1d(kernel_size)
        self.encoder1 = TransformerEncoder(seq_length, feature_size, key_size, 
            query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
            ffn_num_hiddens, num_heads, num_layers, dropout, attn_bias, **kwargs)
        self.encoder2 = TransformerEncoder(seq_length, feature_size, key_size, 
            query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
            ffn_num_hiddens, num_heads, num_layers, dropout, attn_bias, **kwargs)
        self.Linear = nn.Linear(num_hiddens, num_hiddens)
        

    def forward(self, x, enc_mask=None):
        # Conv to the time dim
        x_1 = (self.maxpool(self.conv(x.permute(0, 2, 1)))).permute(0, 2, 1)
        enc_1 = self.encoder1(x_1, enc_mask)
        enc_2 = self.encoder2(x, enc_mask)
        return enc_1+enc_2


class DualClassifier(nn.Module):
    def __init__(self, seq_length, feature_size, key_size, query_size, 
                 value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout, kernel_size,
                 num_cat, attn_bias=False, conv_bias=False, **kwargs):
        super(DualClassifier, self).__init__(**kwargs)
        self.encoder = DualNetEncoder(seq_length, feature_size, key_size,
            query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
            ffn_num_hiddens, num_heads, num_layers, dropout, kernel_size,
            attn_bias, conv_bias, **kwargs)
        self.conv = nn.Conv1d(seq_length, 1, 1)
        self.out = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_cat)
        )


    def forward(self, x):
        enc_x = self.encoder(x, None)
        conv_x = self.conv(enc_x)
        return self.out(conv_x)