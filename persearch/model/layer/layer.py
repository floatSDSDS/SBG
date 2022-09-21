import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from persearch.utils import neg_sampling


class Tower(nn.Module):
    def __init__(self, dropout, d_inp, d_hidden, d_outp):
        super(Tower, self).__init__()
        self.tower = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_inp, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_outp),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.tower(x)


def linear_inited(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()
    return lin


def init_xavier_uniform(*args):
    for arg in args:
        xavier_uniform_(arg)


class Forward(nn.Module):
    def __init__(self, num_layers=5, dim_input=64, dim_out=1,
                 dim_hidden=[256, 128, 256, 32], activation_func='relu',
                 dropout=0.5):
        super(Forward, self).__init__()
        activation_dict = {'relu': nn.ReLU(inplace=True),
                           'tanh': nn.Tanh()}
        try:
            activation_func = activation_dict[activation_func]
        except:
            raise AssertionError(activation_func + " is not supported yet!")
        assert num_layers == len(dim_hidden) + 1, 'num_layers does not match the length of dim_hidden'

        layers = []
        if num_layers == 1:
            layers = layers + [nn.Linear(dim_input, dim_out)]
        else:
            for idx in range(num_layers - 1):
                layers = layers + [
                    nn.Linear(dim_input if idx == 0 else dim_hidden[idx - 1], dim_hidden[idx]),
                    activation_func,
                    nn.Dropout(dropout)]
            layers = layers + [nn.Linear(dim_hidden[-1], dim_out)]

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.seq(x)
        return logits


class SimpleLayer(nn.Module):
    def __init__(self, dim_input, dim_out, dropout):
        super(SimpleLayer, self).__init__()
        self.layer = nn.Sequential(
            linear_inited(dim_input, dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layer(x)


class ParagraphVecLayer(nn.Module):
    """ light implementation of PV-DBOW as claimed in [1]
    (didn't use hierarchical softmax)
    [1] Learning a hierarchical embedding model for personalized product search
    ------
    instead of initialize embeddings for terms and docs, this class requires
    initialization at first place, and return language model loss term
    """
    def __init__(self, emb_entity, emb_term, distribution, k=5, subtract=True):
        super(ParagraphVecLayer, self).__init__()
        self.emb_e = emb_entity  # nn.Embedding layer  [n_e, dim]
        self.emb_v = emb_term  # nn.Embedding Layer  [n_v, dim]
        self.distribution = distribution  # T[n_v]

        assert self.emb_e.embedding_dim == self.emb_v.embedding_dim
        assert self.emb_v.num_embeddings == self.distribution.shape[0]

        self.n_v = self.emb_v.num_embeddings
        self.k = k  # generate k negative samples for each positive token
        self.subtract = subtract
        # self.log_sigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids_entity, token_pos):
        """
        :param ids_entity: T[n], ids corresponding to self.emb_entity
        :param token_pos: T[n, pad_len], actual tokens in the entity,
            corresponding to token_pos
        :return: loss
        """
        bsz, max_len = token_pos.shape
        lens = (token_pos != 0).sum(dim=1)
        n_token = (self.k + 1) * lens.sum()
        token_neg = neg_sampling(token_pos, n_vocab=self.n_v, n_neg=max_len * self.k,
                                 weight=self.distribution, subtract=False, replace=True)
        # print('index token: {}-{}/{}'.format(token_pos.max(), token_neg.max(), self.n_v))
        emb_v_pos = self.emb_v(token_pos)
        emb_v_neg = self.emb_v(token_neg)
        emb_v = torch.cat((emb_v_pos, emb_v_neg), dim=1)  # [bsz, max_len(k+1), dim]

        # print('index entity: {}/{}'.format(ids_entity.max(), self.emb_e.num_embeddings))
        emb_e = self.emb_e(ids_entity)  # [bsz, dim]
        emb_expand = emb_e.unsqueeze(1).expand(bsz, max_len * (self.k + 1), -1)
        dot = emb_expand.mul(emb_v).sum(-1)
        # loss = - self.log_sigmoid(dot).sum() / n_token
        loss = - torch.log(self.sigmoid(dot) + 0.5).sum() / n_token
        return loss


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert not indices.requires_grad
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    @staticmethod
    def forward(indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=2,
                 dropout=0.5, bidirectional=True):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, bidirectional=bidirectional)

    def forward(self, emb_inp, lens):  # [bsz, max_len, d_inp], [bsz]
        emb_pack = pack_padded_sequence(emb_inp, lens, batch_first=True, enforce_sorted=False)
        emb_ctx, _ = self.lstm(emb_pack)
        emb_out, lens_out = pad_packed_sequence(emb_ctx, batch_first=True)
        return emb_out  # [bsz, max_len, 2d]


class SelfAttLayer(nn.Module):
    def __init__(self, d_ctx, d_att, dropout=0.5, dim=1):
        super(SelfAttLayer, self).__init__()
        self.layer_att = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_ctx, d_att),
            nn.Tanh(),
            nn.Linear(d_att, 1),
            nn.Softmax(dim=dim),
        )

    def forward(self, emb_ctx):  # [bsz, T, *, d_ctx]
        attention = self.layer_att(emb_ctx)  # [bsz, T, *, 1]
        emb_att = torch.mul(emb_ctx, attention)  # [bsz, T, *, d_ctx]
        emb_aggregate = torch.sum(emb_att, dim=1)  # [bsz, *, d_ctx]
        return emb_aggregate


class SelfAttEncoder(nn.Module):
    def __init__(self, d_ctx, d_att):
        super(SelfAttEncoder, self).__init__()
        self.att = SelfAttLayer(d_ctx, d_att)

    def forward(self, ids, tokens, emb_v):
        bsz, pad_len = ids.shape  # [bsz, pad_len]
        ids_flatten = ids.view(-1)  # [bsz * pad_len]
        tokens_flatten = tokens[ids_flatten]

        emb_inp = emb_v(tokens_flatten)
        emb_agg = self.att(emb_inp)
        emb_agg_view = emb_agg.view(bsz, pad_len, -1)  # [bsz, pad_len, d]
        return emb_agg_view


class AvgEncoder(nn.Module):
    def __init__(self, dim=-2):
        super(AvgEncoder, self).__init__()

    def forward(self, ids, tokens, lens, emb_v):
        """ids: [bsz, n], tokens: [n_i, pad_len], lens: [n_i], emb_v: emb(n_i, d)"""
        tokens_ = tokens[ids]  # [bsz, n, pad_len]
        lens_ = lens[ids]  # [bsz, n]
        emb = emb_v(tokens_)  # [bsz, n, pad_len, d]
        bsz, n, pad_len, d = emb.shape
        emb_sum = torch.sum(emb, dim=2)
        lens_view = lens_.unsqueeze(2).expand(bsz, n, d)
        emb_avg = emb_sum / lens_view
        return emb_avg


class SelfAttLstmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=2,
                 dropout=0.5, bidirectional=True, d_att=16):
        super(SelfAttLstmEncoder, self).__init__()
        self.lstm = LSTMLayer(input_size, hidden_size, num_layers,
                              dropout=dropout, bidirectional=bidirectional)
        self.att = SelfAttLayer(
            2 * hidden_size if bidirectional else hidden_size, d_att)

    def forward(self, ids, tokens, lens, emb_v):
        """
        :param ids: T[n, pad_len]
        :param tokens: T[n_i, pad_len]
        :param lens: T[n_i]
        :param emb_v: T[n_i, d]
        :return:
        """
        token_batch = tokens[ids]
        lens_batch = lens[ids]

        emb = emb_v(token_batch)
        emb_lstm = self.lstm(emb, lens_batch)  # [bsz, batch_pad_len, 2*d_h]
        emb_att = self.att(emb_lstm)
        return emb_att


class AttZAM(nn.Module):
    def __init__(self, d, d_attn):
        super(AttZAM, self).__init__()

        self.d = d
        self.d_attn = d_attn

        self.w_f = nn.Parameter(torch.rand(self.d, self.d_attn, self.d))
        self.b_f = nn.Parameter(torch.rand(self.d, self.d_attn))
        self.w_h = nn.Parameter(torch.rand(self.d_attn, 1))
        init_xavier_uniform(self.w_f, self.b_f, self.w_h)
        self.tanh = nn.Tanh()

    def forward(self, emb_q, emb_iseq):
        """f(q, i), e.q. (10)
        :param emb_q: [bsz, d]
        :param emb_pre_seq: [bsz, max_len, d]
        :return weight_attn: [bsz, max_len]
        """
        emb_q_view = emb_q.unsqueeze(1).unsqueeze(-1)  # [bsz, 1, d, 1]
        trans_q = self.w_f.matmul(emb_q_view).squeeze(-1)  # [bsz, d, d_attn]
        bias_q = self.b_f.unsqueeze(0).expand_as(trans_q)  # [bsz, d, d_attn]
        h_q_att = self.tanh(trans_q + bias_q)  # [bsz, d, d_attn]
        h_q_att_view = h_q_att.unsqueeze(1)  # [bsz, 1, d, d_attn]
        emb_pre_seq_view = emb_iseq.unsqueeze(2)  # [bsz, max_len, 1, d]
        h_iq_att = emb_pre_seq_view.matmul(h_q_att_view)  # [bsz, max_len, 1, d_attn]
        w_h_view = self.w_h.unsqueeze(0).unsqueeze(0)  # [1, 1, d_attn, 1]
        weight_attn = h_iq_att.matmul(w_h_view)  # [bsz, max_len, 1, 1]
        return weight_attn.squeeze(-1).squeeze(-1)
