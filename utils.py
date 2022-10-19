import math
import torch
from torch import nn
import torch.nn.functional as F


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)

    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    # masked softmax is used in case that some elements are masked
    # masked elements won't contribute to the output
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        # if not specified, all batch use the same mask
        # generate the mask through
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        # else use specific mask for each batch
        else:
            valid_lens = valid_lens.reshape(-1)

        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


def transpose_qkv(X, num_heads):
    # X has shape (batch size, sequence length, num hiddens)
    # divide last last dimension into h heads
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    
    # now X has shape (batch size, sequence length, num heads, hiddens)
    # move the num heads to the second dimension
    X = X.permute(0, 2, 1, 3)

    # combine the heads and batch size
    # return X with shape (batch size*num heads, sequence length, hiddens)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    # inverse transpose_qkv
    # X has shape (batch size*num heads, sequence length, hiddens)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)

    # return X with shape (batch size*num heads, sequence length, num hiddens)
    # where num hiddens = num heads*hiddens
    return X.reshape(X.shape[0], X.shape[1], -1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2))/math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, 
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    
    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, 
                                                 repeats = self.num_heads,
                                                 dim=0)

        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class CasualConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 dilation=1, bias=True):
        super(CasualConv1d, self).__init__(in_channels, out_channels, 
            kernel_size=kernel_size, stride=stride, dilation=dilation,
            groups=in_channels, bias=bias)

        self.__padding = (kernel_size-1)*dilation

    def forward(self, input):
        out = super(CasualConv1d, self).forward(F.pad(input, (self.__padding, 0)))
        return torch.tanh(out)


class CasualMaxPool1d(nn.MaxPool1d):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super(CasualMaxPool1d, self).__init__(kernel_size, stride, padding, dilation)

        self.__padding = (kernel_size-1)*dilation

    def forward(self, input):
        out = super(CasualMaxPool1d, self).forward(F.pad(input, (self.__padding, 0)))
        return out


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, 
                 ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.activation(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        # Y is output of X entering a block
        return self.ln(self.dropout(Y)+X)


class PositionalEncoding(nn.Module):

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class LearnablePosEncoding(nn.Module):

    def __init__(self, num_hiddens, seq_length):
        super(LearnablePosEncoding, self).__init__()
        self.params = nn.Parameter(torch.randn(1, seq_length, num_hiddens))

    def forward(self, X):
        X = X + self.params[:, :X.shape[1], :].to(X.device)
        return X


class MaskedWeightedMSELoss(nn.Module):
    def __init__(self, lamda=0.5, reduction='mean'):
        super(MaskedWeightedMSELoss, self).__init__()
        self.lamda = lamda
        self.MSELoss1 = nn.MSELoss(reduction=reduction)
        self.MSELoss2 = nn.MSELoss(reduction=reduction)

    
    def forward(self, y_pred, y_true, mask):
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)
        unmasked_pred = torch.masked_select(y_pred, ~mask)
        unmasked_true = torch.masked_select(y_true, ~mask)
        masked_loss = self.MSELoss1(masked_pred, masked_true)
        unmasked_loss = self.MSELoss2(unmasked_pred, unmasked_true)
        return (masked_loss + self.lamda*unmasked_loss)/(1+self.lamda)