import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .layer.layer import AttZAM
from persearch.model.base import Base
from persearch.model.layer.layer import AvgEncoder
from persearch.model.layer.layer_lm import ParagraphVectorLoss
import persearch.model.layer.layer_graph as lg
from persearch.utils import smart_sort
import persearch.gens as gens


class SBG(Base):

    def __init__(self, data, cfg_model):
        super(SBG, self).__init__(data, cfg_model)
        self.user_as_item = cfg_model['user_as_item'],
        self.share_term_emb = cfg_model['share_term_emb'],
        self.neg_ratio_i = cfg_model['neg_ratio_i']
        self.neg_ratio_w = cfg_model['neg_ratio_w']
        self.lambda_uq = cfg_model['lambda_uq']
        self.a_bce = cfg_model['a_bce']
        self.a_lm = cfg_model['a_lm']
        self.d_attn = cfg_model['d_attn']
        self.conv_u = cfg_model['conv_u']
        self.conv_pv = cfg_model['conv_pv']
        self.conv_i = cfg_model['conv_i']

        self.n_u = self.generator.n_u
        self.n_i = self.generator.n_i
        self.get_attn_weight = AttZAM(self.d, self.d_attn)
        self.u_ipad = self.generator.get_u_ipad(data.ind_tr, self.n_u)
        self.key_zero = nn.Embedding(1, self.d)


        try:
            self.distribution = self.generator.get_unigram_distribution(
                data.ind_tr, tokens=self.tokens_i, n_v=self.n_v,
            )
        except:
            self.distribution = self.generator.get_unigram_distribution(
                data.ind_tr, tokens=self.tokens_r, n_v=self.n_v,
            )
        self.term_i = self.emb_v
        self.term_u = self.term_i if self.share_term_emb \
            else nn.Embedding(self.n_v, self.d, padding_idx=0)

        self.key_zero = nn.Embedding(1, self.d)
        self.emb_i = nn.Embedding(self.n_i, self.d, padding_idx=0)
        self.emb_i2 = nn.Embedding(self.n_i, self.d, padding_idx=0)
        self.emb_q = AvgEncoder(dim=-2)
        self.emb_u = nn.Embedding(self.n_u, self.d, padding_idx=0)

        self.genGraph = gens.GenGraphSI(data, gap_len=1, ind=data.ind_tr)  # [for pad and zero-item]
        self.edges = self.genGraph.build_edges()
        self.n_e = self.genGraph.n_e
        self.emb_e = nn.Embedding(self.n_e, self.d, padding_idx=0)
        self.ids_e = torch.tensor(
            range(self.n_e), dtype=torch.long, device=self.tokens_i.device
        )
        self.pv_dbow_u = ParagraphVectorLoss(
            self.term_u, self.distribution, n_neg=self.neg_ratio_w, max_len=data.q90)
        self.pv_dbow_i = ParagraphVectorLoss(
            self.term_i, self.distribution, n_neg=self.neg_ratio_w, max_len=data.q90)

        self.sigmoid = nn.Sigmoid()
        self.loss_bce = nn.BCELoss()

        self.gcn = lg.GcnNet(
            self.d, self.d, a=cfg_model['a_self'],
            drop=cfg_model['drop_gcn'], k=cfg_model['k'])

        self.genGraph = None
        self.edges = None
        self.n_e = None
        self.emb_e = None
        self.ids_e = None

        self.k = cfg_model['k']
        self.genGraph = gens.GenGraphSI(data, gap_len=1, ind=data.ind_tr)  # [for pad and zero-item]
        self.edges = self.genGraph.build_edges()
        self.n_e = self.genGraph.n_e
        self.emb_e = nn.Embedding(self.n_e, self.d, padding_idx=0)
        self.ids_e = torch.tensor(
            range(self.n_e), dtype=torch.long, device=self.tokens_i.device
        )

    def forward(self, uqt, pre_clicks, nids):
        uid, qid = uqt[:, 0], uqt[:, 1]

        emb_all = self.agg_gnn() if 'conv_e' in [self.conv_i, self.conv_u] else None
        cks = self.u_ipad[uid]

        emb_q = self.emb_q(qid.view(-1, 1), self.tokens_q, self.lens_q, self.emb_v)  # [bsz, d]
        emb_q = emb_q.squeeze(1)
        # cks = pre_clicks
        emb_u = self.get_emb_u(emb_q, cks, emb_all)  # [bsz, d]
        emb_uq = self.lambda_uq * emb_q + (1-self.lambda_uq) * emb_u
        emb_uq = emb_uq.unsqueeze(1)

        assert self.conv_i in ['conv_e', 'e', 'i']
        if self.conv_i == 'conv_e':
            emb_i = emb_all[nids]
        if self.conv_i == 'e':
            emb_i = self.emb_e(nids)  # [bsz, pad_len, d]
        if self.conv_i == 'i':
            emb_i = self.emb_i(nids)  # [bsz, pad_len, d]

        sim_uq_i = self.f_sim(emb_uq, emb_i)
        scores = self.sigmoid(sim_uq_i)
        return scores

    def do_train(self, uqt, pre_clicks, nids, target, idx=None):
        scores = self(uqt, pre_clicks, nids)
        pred = self.score_ctr_pred(scores)
        loss = 0.0
        nid_pos = nids[:, 0]
        tokens_pos = [self.generator.tokens_r[i] for i in idx]
        tokens_pos = pad_sequence(tokens_pos, batch_first=True).to(self.device)
        loss += self.a_lm * self.loss_lm(nid_pos, tokens_pos)

        nid_all = nids.flatten()
        loss += (
                self.a_bce * self.loss_bce(scores, target)
                + self.a_lm * self.loss_lm(nid_all)
                # + self.a_lm * self.loss_lm(nid_pos)
        )
        return loss, scores, pred

    def get_emb_u(self, emb_q, pre_clicks, emb_all=None):
        """simple version of aggregation"""

        assert self.conv_u in ['conv_e', 'e', 'i']
        if self.conv_u == 'conv_e':
            emb_pre_seq = emb_all[pre_clicks]
        if self.conv_u == 'e':
            emb_pre_seq = self.emb_e(pre_clicks)
        if self.conv_u == 'i':
            emb_pre_seq = self.emb_i(pre_clicks)

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

        if self.conv_u == 'conv_e':
            emb_pre_seq_zero = emb_all[pre_clicks_zero]# [bsz, max_len+1, d]
        if self.conv_u == 'e':
            emb_pre_seq_zero = self.emb_e(pre_clicks_zero)# [bsz, max_len+1, d]
        if self.conv_u == 'i':
            emb_pre_seq_zero = self.emb_i(pre_clicks_zero)# [bsz, max_len+1, d]

        h_u = emb_pre_seq_zero.mul(attn.unsqueeze(-1)).sum(1)  # [bsz, d]
        return h_u

    def loss_lm(self, nid, tokens=None):

        assert self.conv_pv in ['conv_e', 'e', 'i']
        if self.conv_pv == 'conv_e':
            emb_all = self.agg_gnn()
            emb_i = emb_all[nid]
        if self.conv_pv == 'e':
            emb_i = self.emb_e(nid)
        if self.conv_pv == 'i':
            emb_i = self.emb_i(nid)

        token_i = self.tokens_i[nid] if tokens is None else tokens  # [bsz, max_len]
        loss_lm_i = self.pv_dbow_i(emb_i, token_i)
        return loss_lm_i.sum()

    def agg_gnn(self, ids=None):
        if ids is None:
            emb_e = self.emb_e(self.ids_e).squeeze(1)
            emb_e = self.gcn(emb_e, self.edges.t())
        else:
            emb_e = self.emb_e(ids)
        return emb_e
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
        super(SBG, self).move_to(device)
        self.u_ipad = self.u_ipad.to(device)
        self.edges = self.edges.to(device)
        self.ids_e = self.ids_e.to(device)

