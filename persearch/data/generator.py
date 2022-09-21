from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from numpy.random import choice
import pandas as pd

from persearch.utils.tool import neg_sampling


class DataGenerator(object):
    """ generate data from DataLoader for further usage
    method: get_<>(data, ind, params), return data (in tensor)
    """
    def __init__(self, data, ind=None, label_col='click', device=None):
        self.name = 'Gen'
        self.trans = data.trans.loc[data.ind_sel]
        self.ind = data.ind_sel if ind is None else ind

        # data.user/query/item ids are 1-based, and do not include padded entity
        self.n_u = data.user.shape[0] + 1
        self.n_q = data.query.shape[0] + 1
        self.n_i = data.item.shape[0] + 1  # add padded item
        self.n_v = data.n_v

        self.label_col = label_col

        # just for sampling
        device_available = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')
        self.device = device_available if device is None else device

    def __call__(self, *args, ind=None, **kwargs):
        """custom pipeline for data generation"""

    def get_uqiltt(self, ind, label_col=None, filter_zero=False, pos_only=False):
        """ck_time will be set to -1 if it is nan
        :return uqiltt: T[T, 6], [uid, qid, nid, label, pv_time, ck_time]
        """
        label_col = self.label_col if label_col is None else label_col
        data = self._filter_data(ind, label_col, filter_zero, pos_only)
        uqiltt = data[['uid', 'qid', 'nid', label_col, 'pv_time', 'ck_time']].copy()
        uqiltt = torch.tensor(uqiltt.values).long()
        return uqiltt

    def get_uqiltt_neg_rand(self, ind, n_neg=2, label_col=None, pool=None, weight=None):
        """
        1. get pos data in trans[ind]
        2. couple <n_neg> times random negative samples for each (u-q-i) tuple
            - neg samples are sampled from the whole pool (nids)
            - if pool is none, sample from unique nids in current batch subset
        3.
        :return uqiltt: T[T, 6], [uid, qid, nid, label, pv_time, ck_time]
        """
        uqiltt_pos, negs = self.get_tuple_negs_rand(
            ind, n_neg=n_neg, label_col=label_col, pool=pool, weight=weight)
        uq = uqiltt_pos[:, :2]
        times = uqiltt_pos[:, 4:]
        neg_label = torch.zeros_like(uq[:, 0]).view(-1, 1)
        uqiltt_collect = [uqiltt_pos]
        for i in range(n_neg):
            neg_i = negs[:, i].view(-1, 1)
            uq_neg_l_tt = torch.cat((uq, neg_i, neg_label, times), dim=1)
            uqiltt_collect.append(uq_neg_l_tt)
        uqiltt = torch.cat(uqiltt_collect)
        return uqiltt

    def get_uqiitt_rand(self, ind, n_neg=2, label_col=None, pool=None, weight=None):
        """
        build pos-neg item pair based on label_col in the input batch
        :return uqii: T[n, 4](uid, qid, nid_pos, nid_neg), tt: T[n, 2](pv_time, ck_time)
        """
        uqiltt_pos, negs = self.get_tuple_negs_rand(
            ind, n_neg=n_neg, label_col=label_col, pool=pool, weight=weight)
        uqi = uqiltt_pos[:, :3]
        tt = uqiltt_pos[:, 4:]
        uqiitt_collect = []
        for i in range(n_neg):
            neg_i = negs[:, i].view(-1, 1)
            uqi_neg_tt = torch.cat((uqi, neg_i, tt), dim=1)
            uqiitt_collect.append(uqi_neg_tt)
        uqiitt = torch.cat(uqiitt_collect)
        return uqiitt

    def get_uqiitt_on_uq(self, ind, n_neg=1, label_col=None):
        """ (better not use with batch index)
        build pos-neg item pair based on label_col (by user)
        transaction should have cols [nid, uid, qid] for item, user, and query
        :param ind: list lf index correspond to self.data_raw
        :param n_neg: int, ratio of negative sample
        :param label_col: select col name to be label
        :return: LongTensor[n, 6] [uid, qid, nid_pos, nid_neg, pv_time, ck_time]
        """
        label_col = self.label_col if label_col is None else label_col
        data = self.trans.loc[ind, ['uid', 'qid', 'nid', label_col, 'pv_time', 'ck_time']]
        group_uq = data.groupby(['uid', 'qid'])
        group_agg = group_uq.agg(
            n_pos=(label_col, np.sum),
            freq=(label_col, len)
        )
        group_agg['p_pos'] = group_agg['n_pos'] / group_agg['freq']
        uq_keep = group_agg[group_agg['p_pos'] > 0]
        uq_keep = uq_keep[uq_keep['p_pos'] < 1]
        uq_keep = uq_keep.groupby(['uid', 'qid'])

        make_data = []
        for (u, q), v in tqdm(uq_keep):
            raw = group_uq.get_group((u, q))
            raw_pos = raw.loc[raw[label_col] > 0]
            nid_pos = raw_pos['nid']
            pv_time = raw_pos['pv_time']
            ck_time = raw_pos['ck_time']
            neg_candidate = raw.loc[raw[label_col] <= 0, 'nid']
            pos_bag = [nid_pos for i in range(n_neg)]
            pos_dup = pd.concat(pos_bag)
            neg = np.random.choice(neg_candidate, size=pos_dup.shape[0], replace=True)
            sample = pd.DataFrame(dict(
                uid=u, qid=q, pos=pos_dup, neg=neg, pv_time=pv_time, ck_time=ck_time
            ))
            make_data.append(sample)
        data_uqii_on_uq = pd.concat(make_data)
        data_uqii_on_uq = torch.tensor(data_uqii_on_uq.to_numpy()).long()
        return data_uqii_on_uq

    def get_topk_mix_tuple(self, ind, pool=None, label_col=None, topk=100):
        """
        1. get pos data in self.trans[ind]
        2. for each tuple, pad neg sample from pool to topk
            return uqtti and mix T[n, topk] (pad <topk-1> neg samples)
        """
        pool = self.trans['nid'].unique() if pool is None else pool
        label_col = self.label_col if label_col is None else label_col
        data = self.trans.loc[ind]
        data_pos = data[data[label_col] > 0]
        uqtti = torch.tensor(data_pos[['uid', 'qid', 'pv_time', 'ck_time', 'nid']].values)

        target_nid = data_pos['nid'].values.reshape((-1, 1))
        n_neg_sample = len(target_nid) * (topk - 1)
        neg_nid = choice(pool, n_neg_sample).reshape((-1, (topk - 1)))
        while True:
            ind_resample = np.where((neg_nid - target_nid) == 0)
            n_resample = ind_resample[0].shape[0]
            if n_resample == 0:
                break
            neg_nid[ind_resample[0], ind_resample[1]] = choice(pool, n_resample)

        topk_mix = np.concatenate([target_nid, neg_nid], axis=1)
        topk_mix = torch.tensor(topk_mix)
        return uqtti, topk_mix

    def get_topk_mix_uq(self, ind, pool=None, label_col=None, topk=100):
        """
        1. get pos data in self.trans[ind], use the uq_time(min positive time) as t
        2. for each uq pair, pad it to length topk with negative samples
        :return uq_test: T[n, 2]
        :return pad_nids: T[n, pad_len]
        :return topk_mix: T[n, topk]
        """
        pool = torch.tensor(self.trans['nid'].unique()) if pool is None else pool
        uqt, nids_pad = self.get_uqt_pos_ipad(ind, label_col)
        nids_neg = neg_sampling(nids_pad, self.n_i, n_neg=topk, pool=pool)
        n = uqt.shape[0]
        n_pad_topk = topk - nids_pad.shape[1]
        pad_right = torch.zeros((n, n_pad_topk), dtype=torch.long)
        nids_pad_topk = torch.cat((nids_pad, pad_right), dim=1)
        mask_pad = (nids_pad_topk == 0).int()
        nids_mix = nids_neg.mul(mask_pad) + nids_pad_topk
        return uqt, nids_pad, nids_mix

    def get_topk_mix_uqi(self, ind, pool=None, label_col=None, topk=100):
        """
        1. get pos data in self.trans[ind], use the uq_time(min positive time) as t
        2. for each uq pair, pad it to length topk with negative samples
        :return uq_test: T[n, 2]
        :return pad_nids: T[n, pad_len]
        :return topk_mix: T[n, topk]
        """
        pool = torch.tensor(self.trans['nid'].unique()) if pool is None else pool
        data = self.trans.loc[ind, :]
        label_col = self.label_col if label_col is None else label_col
        uqi_pos = data[data[label_col] > 0]
        uqit = uqi_pos[['uid', 'qid', 'nid', 'pv_time']]
        # uqit = torch.tensor(uqit.values, dtype=torch.long)
        nids_pad = torch.tensor(uqit.iloc[:, 2].values, dtype=torch.long).view(-1, 1)
        # uqt, nids_pad = self.get_uqt_pos_ipad(ind, label_col)
        nids_neg = neg_sampling(nids_pad, self.n_i, n_neg=topk, pool=pool)
        n = uqit.shape[0]
        n_pad_topk = topk - nids_pad.shape[1]
        pad_right = torch.zeros((n, n_pad_topk), dtype=torch.long)
        nids_pad_topk = torch.cat((nids_pad, pad_right), dim=1)
        mask_pad = (nids_pad_topk == 0).int()
        nids_mix = nids_neg.mul(mask_pad) + nids_pad_topk
        return uqit, nids_pad, nids_mix

    def get_tuple_negs_rand(self, ind, n_neg=2, label_col=None, pool=None, weight=None):
        """
        1. get pos data in trans[ind]
        2. return uqil and corresponding negative samples (nid, item ids)
        :param ind: list of ind corresponding to pd.index
        :param n_neg: int
        :param label_col: str, indicate column name of the data
        :param pool: T[n_pool], item candidate to generate negative samples
                (assume nid in pool is unique)
        :param weight: T[n_pool], indicate weight of item
                (corresponding to item in pool by position)
        :return: uqiltt: T[n_pos, 6], negs: T[n_pos, n_neg]
        """
        label_col = self.label_col if label_col is None else label_col
        data = self.trans.loc[ind, :]
        data_pos = data[data[label_col] > 0][['uid', 'qid', 'nid', label_col, 'pv_time', 'ck_time']]
        uqiltt_pos = torch.tensor(data_pos.values).long()
        nids = uqiltt_pos[:, 2].view(-1, 1)

        pool_ = self.trans['nid'].unique() if pool is None else pool.clone()
        pool_ = torch.tensor(pool_)
        weight_ = torch.ones_like(pool_) if weight is None else weight.clone()
        weight_ = weight_.float()

        negs = neg_sampling(nids, n_vocab=self.n_i, n_neg=n_neg,
                            pool=pool_, weight=weight_)
        return uqiltt_pos, negs

    def get_uqt_pos_ipad(self, ind, label_col=None, idx_pad=0, max_len=20):
        """todo: max_len
        for each uq pair, return a padded positive item set (not behavior seq)
        :param ind: [index], a list of index correspond to self.trans
        :param label_col: str, column name in self.trans as label
        :param idx_pad: int, pad item index with idx_pad
        :param max_len: int, if there are more than <max_len> item, sample max_len of them
        :return uqt T[n_uq, 3], nids T[n_uq, pad_len]
        """
        label_col = self.label_col if label_col is None else label_col
        data = self.trans.loc[ind, :]
        uqi_group_pos = data[data[label_col] > 0]
        uqt_i = uqi_group_pos.groupby(['uid', 'qid'])[['nid', 'pv_time']].agg(
            uq_time=('pv_time', min),
            iset=('nid', set)
        )
        uqt = uqt_i.reset_index()[['uid', 'qid', 'uq_time']]
        uqt = torch.tensor(uqt.values, dtype=torch.long)

        iset = [torch.tensor(list(x)) for x in uqt_i['iset']]
        nids = pad_sequence(iset, batch_first=True, padding_value=idx_pad)
        return uqt, nids

    def get_u_ipad(self, ind, n_u, label_col=None, idx_pad=0):
        """return T[n_u(all), pad_pos_item]"""
        print('create padded item history for user')
        label_col = self.label_col if label_col is None else label_col
        data = self.trans.loc[ind, :]
        ui_group_pos = data[data[label_col] > 0]
        u_i = ui_group_pos.groupby('uid')[['nid']].apply(
            lambda x: torch.tensor(np.unique(x)))
        ui_lst = [torch.zeros(1, dtype=torch.long)] * (n_u + 1)
        for u, nid in tqdm(u_i.iteritems()):
            ui_lst[u] = torch.cat([ui_lst[u], nid])
        u_ipad = pad_sequence(ui_lst, batch_first=True)
        return u_ipad

    def get_q_ipad(self, ind, label_col=None, idx_pad=0):
        """
        :return qids T[n_q, 1], nids T[n_q, pad_len]
        """
        label_col = self.label_col if label_col is None else label_col
        data = self.trans.loc[ind, :]
        qi_group_pos = data[data[label_col] > 0]
        q_i = qi_group_pos.groupby('qid')['nid'].apply(
            lambda x: torch.tensor(np.unique(x)))
        qids = q_i.reset_index()[['qid']]
        qids = torch.tensor(qids.values, dtype=torch.long)
        nids = pad_sequence(q_i.tolist(), batch_first=True, padding_value=idx_pad)
        return qids, nids

    def get_ui_matrix(self, ind, label_col='click'):
        """
        return sparse ui matrix within index (correspond to data.data row index)
        :param ind: [data.data index]
        :param label_col: str, selected col to fill the sparse matrix
        :return: ui_matrix: sparse tensor[n_u, n_i]
        """
        data = self.trans.loc[ind, :]
        data_nonzero = data[data[label_col] != 0]

        val_fill = torch.tensor(data_nonzero[label_col].to_numpy()).float()

        ui_adj_list = data_nonzero[['uid', 'nid']]
        ui_adj_list = torch.tensor(ui_adj_list.to_numpy())

        ui_matrix = torch.sparse.FloatTensor(ui_adj_list.t(), val_fill,
                                             torch.Size([self.n_u, self.n_i]))
        return ui_matrix

    def get_qi_matrix(self, ind, label_col='click'):
        """
        return sparse qi matrix within index (correspond to data.data row index)
        :param ind: [data.dtat index]
        :param label_col: str, selected col to fill the sparse matrix
        :return: qi_matrix: sparse tensor[n_q, n_i]
        """
        data = self.trans.loc[ind, :]
        data_pos = data[data[label_col] != 0]

        val_fill = torch.tensor(data_pos[label_col].to_numpy()).float()

        qi_adj_list = data_pos[['qid', 'nid']]
        qi_adj_list = torch.tensor(qi_adj_list.to_numpy())

        qi_matrix = torch.sparse.FloatTensor(qi_adj_list.t(), val_fill,
                                             torch.Size([self.n_q, self.n_i]))
        return qi_matrix

    def get_unigram_distribution(self, ind, tokens, n_v, idx_pad=0,
                                 distortion=0.75, label_col='click'):
        """
        return unigram distribution as claimed in HEM (commonly used in word2vec)
        assume the padded token in tokens is 0, the returned distribution will
        place 0 in the first place distribution[0]=1
        :param ind: list of index, selected index in self.trans
        :param tokens: T[n_doc, pad_len], padded tokenized docs (row corresponding to nid)
        :param n_v: int, size of vocabulary
        :param idx_pad: 0
        :param distortion: float, power of the frequency
        :param label_col: column in self.data work as label
        :return: distribution T[n_w], pdf on vocabulary
        """
        vocab_freq = torch.zeros(n_v, dtype=torch.long)
        data = self.trans.loc[ind, ['nid', label_col]]
        nid_pos = data[data[label_col] > 0]['nid'].to_numpy()
        # remove padded and calculate frequency
        token_corpus = tokens[nid_pos]
        w_id, w_freq = token_corpus.unique(sorted=True, return_counts=True)
        w_freq[w_id == idx_pad] = 0
        vocab_freq = vocab_freq.scatter(dim=0, index=w_id, src=w_freq)
        # assume each term has at least one appearance
        # (then it can be sampled in negative sampling)
        vocab_freq = vocab_freq + 1
        vocab_freq_pow = vocab_freq.float().pow(distortion)
        sum_vocab_freq = vocab_freq_pow.sum()
        vocab_distribute = vocab_freq_pow / sum_vocab_freq
        return vocab_distribute

    def get_cki(self, ind, by='sid'):
        """
        :return ids: array[n_unit]
        :return iseq: array[[seq_u1],[seq_u2],...[seq_un]]
        :return tseq: array[[seq_u1],[seq_u2],...[seq_un]]
        """
        data = self.trans.loc[ind, :]
        data_pos = data[data['click'] > 0].copy()
        data_pos = data_pos[[by, 'ck_time', 'nid']]
        sorted_ck_group = data_pos.groupby(by).apply(
            lambda x: x.sort_values('ck_time', ascending=True)).reset_index(drop=True)
        uid_iseq = sorted_ck_group.groupby(by)['nid'].apply(
            lambda x: x.values).reset_index()
        uid_itime = sorted_ck_group.groupby(by)['ck_time'].apply(
            lambda x: x.values).reset_index()

        ids = np.array(uid_iseq[by].tolist())
        iseqs = np.array(uid_iseq['nid'].tolist())
        tseqs = np.array(uid_itime['ck_time'].tolist())
        return ids, iseqs, tseqs

    def get_u_n_iseq_itime(self, label_col=None):
        """

        :return uids L[n_u, 1],nids L[n_u,1], iseqs L[[seq_u1],[seq_u2],...[seq_un]]
        """
        label_col = self.label_col if label_col is None else label_col
        data = self.trans
        uic_group_pos = data[data[label_col] > 0]
        u_i_ck = uic_group_pos[['uid', 'nid', 'ck_time']]
        sorted_ck_group = u_i_ck.groupby('uid').apply(
            lambda x: x.sort_values('ck_time', ascending=True)).reset_index(drop=True)
        uid_iseq = sorted_ck_group.groupby('uid')['nid'].apply(
            lambda x: x.tolist()).reset_index()
        uid_itime = sorted_ck_group.groupby('uid')['ck_time'].apply(
            lambda x: x.tolist()).reset_index()
        uids = np.array(uid_iseq['uid'].tolist())
        iseqs = np.array(uid_iseq['nid'].tolist())
        itime = np.array(uid_itime['ck_time'].tolist())
        # nids = np.array(list(set(
        #     uic_group_pos['nid'].tolist())))  # base on the use's clicked item
        nids = np.array(list(set(
            self.trans_all['nid'].tolist()
        )))
        return uids, nids, iseqs, itime

    def _filter_data(self, ind, label_col, filter_zero=False, pos_only=False):
        data = self.trans.loc[ind, :]
        if filter_zero:
            data = data.loc[data[label_col] != 0, :]
        if pos_only:
            data = data.loc[data[label_col] > 0]
        return data

    def get_qseq_iseq_iseq(self, ind, label_col=None):
        """
        1. get click data
        2. return sorted query_sequence, item_sequence, and corresponding time
            for each user
        """
        label_col = self.label_col if label_col is None else label_col
        data = self.trans.loc[ind, :]
        uic_group_pos = data[data[label_col] > 0]
        uq_i_ck = uic_group_pos[['uid', 'qid', 'nid', 'ck_time']]
        u_df = uq_i_ck.groupby(['uid'])
        qseq = dict().fromkeys(range(self.n_u), np.zeros(1))
        iseq = dict().fromkeys(range(self.n_u), np.zeros(1))
        tseq = dict().fromkeys(range(self.n_u), np.zeros(1))
        for uid, df in tqdm(u_df):
            sub_df = df.sort_values(by='ck_time', ascending=True)
            qseq_u = np.array(sub_df['qid'].tolist())
            iseq_u = np.array(sub_df['nid'].tolist())
            tseq_u = np.array(sub_df['ck_time'].tolist())
            qseq[uid] = np.concatenate([qseq[uid], qseq_u])
            iseq[uid] = np.concatenate([iseq[uid], iseq_u])
            tseq[uid] = np.concatenate([tseq[uid], tseq_u])
        return qseq, iseq, tseq

    def get_u_qt_iseqt(self, label_col=None):
        label_col = self.label_col if label_col is None else label_col
        data = self.trans
        uic_group_pos = data[data[label_col] > 0]
        uq_i_ck = uic_group_pos[['uid', 'qid', 'nid', 'pv_time', 'ck_time']]
        uids = []
        qids = []
        nid_ts = []
        u_df = uq_i_ck.groupby(['uid'])
        for name, df in tqdm(u_df):
            uids.append((df['uid'].tolist()[0]))
            temp_nid_t = []
            temp_qt = []
            for sub_name, sub in df.groupby(['uid', 'qid']):
                sub_v = sub.sort_values(by='pv_time', ascending=True)
                temp_qt.append((int(sub_v.iloc[0]['qid']), sub_v.iloc[0]['pv_time']))
                sub_c = sub.sort_values(by='ck_time', ascending=True)
                nid = sub_c['nid'].tolist()
                ck_t = sub_c['ck_time'].tolist()
                data = [(i, j) for i, j in zip(nid, ck_t)]
                temp_nid_t.append(data)
            nid_ts.append(temp_nid_t)
            qids.append(temp_qt)
        return uids, qids, nid_ts

    def get_uqitl(self, ind, label_col=None, filter_zero=False, pos_only=False):
        """"""
        label_col = self.label_col if label_col is None else label_col
        data = self.trans.loc[ind, :]
        if filter_zero:
            data = data.loc[data[label_col] != 0, :]
        if pos_only:
            data = data.loc[data[label_col] > 0]
        uqitl = data[['uid', 'qid', 'nid', 'pv_time', label_col]].values
        return uqitl
