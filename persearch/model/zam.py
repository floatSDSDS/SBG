import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from persearch.model.base import Base
from .layer.layer import AttZAM
from persearch.utils import smart_sort
from persearch.model.layer.layer import AvgEncoder
from persearch.model.layer.layer_lm import ParagraphVectorLoss


class AEM(Base):
    """ implementation of AEM
    to use this model, ensure each user will present in each split
    ---- ref
    Ai, Q., Hill, D. N., Vishwanathan, S. V. N., & Croft, W. B. (2019, November).
    A zero attention model for personalized product search. In Proceedings of
    the 28th ACM International Conference on Information and Knowledge Management
    (pp. 379-388).
    """
    def __init__(self, data, cfg_model):
        super(AEM, self).__init__(data, cfg_model)
        self.user_as_item = cfg_model['user_as_item'],
        self.share_term_emb = cfg_model['share_term_emb'],
        self.neg_ratio_i = cfg_model['neg_ratio_i']
        self.neg_ratio_w = cfg_model['neg_ratio_w']
        self.lambda_uq = cfg_model['lambda_uq']
        self.a_bce = cfg_model['a_bce']
        self.a_lm = cfg_model['a_lm']

        self.n_u = self.generator.n_u
        self.n_i = self.generator.n_i
        try:
            print("tokens_i")
            self.distribution = self.generator.get_unigram_distribution(
                data.ind_tr, tokens=self.tokens_i, n_v=self.n_v,
            )
        except:
            print("tokens_r")
            self.distribution = self.generator.get_unigram_distribution(
                data.ind_tr, tokens=self.tokens_r, n_v=self.n_v,
            )

        self.term_i = self.emb_v
        self.term_u = self.term_i if self.share_term_emb \
            else nn.Embedding(self.n_v, self.d, padding_idx=0)

        self.emb_i = nn.Embedding(self.n_i, self.d, padding_idx=0)
        self.emb_i2 = nn.Embedding(self.n_i, self.d, padding_idx=0)
        self.emb_q = AvgEncoder(dim=-2)
        self.emb_u = nn.Embedding(self.n_u, self.d, padding_idx=0)

        self.pv_dbow_u = ParagraphVectorLoss(
            self.term_u, self.distribution, n_neg=self.neg_ratio_w, max_len=data.q90)
        self.pv_dbow_i = ParagraphVectorLoss(
            self.term_i, self.distribution, n_neg=self.neg_ratio_w, max_len=data.q90)

        self.sigmoid = nn.Sigmoid()
        self.loss_bce = nn.BCELoss()

        self.d_attn = cfg_model['d_attn']
        self.get_attn_weight = AttZAM(self.d, self.d_attn)
        self.u_ipad = self.generator.get_u_ipad(data.ind_tr, self.n_u)

        self.conv_u = cfg_model['conv_u']
        self.conv_pv = cfg_model['conv_pv']
        self.conv_i = cfg_model['conv_i']

    def forward(self, uqt, pre_clicks, nids):
        uid, qid = uqt[:, 0], uqt[:, 1]
        cks = self.u_ipad[uid]

        emb_q = self.emb_q(qid.view(-1, 1), self.tokens_q, self.lens_q, self.emb_v)  # [bsz, d]
        emb_q = emb_q.squeeze(1)
        # cks = pre_clicks
        emb_u = self.get_emb_u(emb_q, cks)  # [bsz, d]
        emb_uq = self.lambda_uq * emb_q + (1-self.lambda_uq) * emb_u
        emb_uq = emb_uq.unsqueeze(1)

        emb_i = self.emb_i2(nids) if self.conv_i == 'i' else self.emb_i(nids)  # [bsz, pad_len, d]
        sim_uq_i = self.f_sim(emb_uq, emb_i)
        scores = self.sigmoid(sim_uq_i)
        return scores

    def do_train(self, uqt, pre_clicks, nids, target, idx=None):
        scores = self(uqt, pre_clicks, nids)
        pred = self.score_ctr_pred(scores)
        loss = 0.0
        try:
            nid_pos = nids[:, 0]
            tokens_pos = [self.generator.tokens_r[i] for i in idx]
            tokens_pos = pad_sequence(tokens_pos, batch_first=True).to(self.device)
            loss += self.a_lm * self.loss_lm(nid_pos, tokens_pos)
        except:
            tokens_pos = None
        nid_all = nids.flatten()
        loss += (
                self.a_bce * self.loss_bce(scores, target)
                + self.a_lm * self.loss_lm(nid_all)
                # + self.a_lm * self.loss_lm(nid_pos)
        )
        return loss, scores, pred

    def get_emb_u(self, emb_q, pre_clicks):
        """simple version of aggregation"""
        emb_pre_seq = self.emb_i(pre_clicks)
        weight_attn = self.get_attn_weight(emb_q, emb_pre_seq)  # [bsz, max_len]
        weight_attn_div_max = weight_attn / weight_attn.abs().max()
        attn_exp = weight_attn_div_max.exp()  # [bsz, max_len]
        mask = (pre_clicks != 0)  # [bsz, max_len]
        attn_exp_masked = attn_exp.mul(mask)  # [bsz, max_len]
        attn_exp_sum = attn_exp_masked.sum(1).unsqueeze(-1)  # [bsz, 1]
        attn = attn_exp_masked / (attn_exp_sum + 1e-3)  # [bsz, max_len]
        h_u = emb_pre_seq.mul(attn.unsqueeze(-1)).sum(1)  # [bsz, d]
        return h_u

    def loss_lm(self, nid, tokens=None):

        emb_i = self.emb_i2(nid) if self.conv_pv == 'i' else self.emb_i(nid)
        token_i = self.tokens_i[nid] if tokens is None else tokens  # [bsz, max_len]
        loss_lm_i = self.pv_dbow_i(emb_i, token_i)
        return loss_lm_i.sum()

    def predict_topk_mix(self, data_test, ind_sel):
        """
        :param data_test: DataLoader(uqtt, mix_nids)
        :param ind_sel: list[item index]
        """
        batch_size = int(self.batch_size / 128)
        uqt, nids = data_test
        nids = nids.to(self.device)
        k = nids.shape[1]
        data_test = self.generator.get_data_test(ind_sel, uqt, nids, batch_size)
        scores = []
        for i_batch, data_batch in enumerate(data_test):
            data_batch = self.batch_to(data_batch)
            # scores_batch = []
            # for ind_k in batch_generator_n(k, self.batch_size):
            #     nids_part = data_batch[2][:, ind_k]
            #     batch_tmp = (data_batch[0], data_batch[1], nids_part)
            #     score_batch_part = self(*batch_tmp)
            #     scores_batch.append(score_batch_part)
            # scores_batch = torch.cat(scores_batch, dim=1)
            scores_batch = self(*data_batch)
            scores.append(scores_batch)
        collect_scores = torch.cat(scores, dim=0)
        nids_rank = collect_scores.topk(k)[1]
        nids_sort = smart_sort(nids, nids_rank)
        return nids_sort

    def move_to(self, device):
        super(AEM, self).move_to(device)
        self.u_ipad = self.u_ipad.to(device)


class ZAM(AEM):
    """ implementation of ZAM
    to use this model, ensure each user will present in each split
    ---- ref
    Ai, Q., Hill, D. N., Vishwanathan, S. V. N., & Croft, W. B. (2019, November).
    A zero attention model for personalized product search. In Proceedings of
    the 28th ACM International Conference on Information and Knowledge Management
    (pp. 379-388).
    """
    def __init__(self, data, cfg_model):
        super(ZAM, self).__init__(data, cfg_model)
        self.key_zero = nn.Embedding(1, self.d)

    def get_emb_u(self, emb_q, pre_clicks):
        """simple version of aggregation"""

        emb_pre_seq = self.emb_i2(pre_clicks) if self.conv_u == 'i' else self.emb_i(pre_clicks)

        zero_item = (pre_clicks[:, 0] != pre_clicks[:, 0]).view(-1, 1).long()
        emb_plus = self.key_zero(zero_item)

        emb_pre_seq_plus = torch.cat([emb_pre_seq, emb_plus], dim=1)

        weight_attn = self.get_attn_weight(emb_q, emb_pre_seq_plus)  # [bsz, max_len]
        weight_attn_div_max = weight_attn / weight_attn.abs().max()
        attn_exp = weight_attn_div_max.exp()  # [bsz, max_len]

        mask_zero = (zero_item == 0)
        mask = torch.cat(((pre_clicks != 0), mask_zero), dim=1)  # [bsz, max_len]
        attn_exp_masked = attn_exp.mul(mask)  # [bsz, max_len]

        attn_exp_sum = attn_exp_masked.sum(1).unsqueeze(-1)  # [bsz, 1]
        attn = attn_exp_masked / (attn_exp_sum + 1e-3)  # [bsz, max_len]

        pre_clicks_zero = torch.cat((pre_clicks, zero_item), dim=1)  # [bsz, max_len+1]
        # emb_pre_seq_zero = self.emb_i(pre_clicks_zero)  # [bsz, max_len+1, d]
        emb_pre_seq_zero = self.emb_i2(pre_clicks_zero) if self.conv_u == 'i' else self.emb_i(pre_clicks_zero)
        h_u = emb_pre_seq_zero.mul(attn.unsqueeze(-1)).sum(1)  # [bsz, d]
        return h_u
