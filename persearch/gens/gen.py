import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from persearch.data.generator import DataGenerator


"""store pre-defined generator pipelines
    __call__(self, *args, ind=None, **kwargs)
    :return torch.DataLoader((uq, nids, target)):
        - uq [n, 2]
        - nids [n, pad_len]
        - target [n, pad_len]
"""


class GetUqilTuple(DataGenerator):
    """ return DataLoader(user_query[n, 2], nids[n, 1], target[n, 1])
    - where there are two strategies for QUIL, real and rand
        - real: call get_uqil, real will return all the uqil with ind
        - rand: call get_uqil_rand, rand will return positive uqi with
        <n_neg> times negative samples sampled from self.trans[ind]
    """
    def __init__(self, data, ind=None,
                 use_real=True, use_rand=True, n_neg=2, batch_size=256,
                 label_col='click'):
        super(GetUqilTuple, self).__init__(data, ind, label_col)
        self.name = 'GetUqilTuple'
        self.use_real = use_real
        self.use_rand = use_rand
        self.n_neg = n_neg
        self.batch_size = batch_size
        assert any([use_real, use_rand])

    def __call__(self, *args, ind=None, **kwargs):
        ind = self.ind if ind is None else ind
        uqil_cat = self.collect_data(ind)
        uq_pair = uqil_cat[:, :2]
        nids = uqil_cat[:, 2].view(-1, 1)  # [n, 1]
        target = uqil_cat[:, 3].float().view(-1, 1)  # [n, 1]
        uqil_zip = [(uq, nid, l) for uq, nid, l in zip(uq_pair, nids, target)]
        uqil_iter = DataLoader(uqil_zip, batch_size=self.batch_size,
                               pin_memory=True, shuffle=True)
        return uqil_iter

    def collect_data(self, ind):
        uqil_lst = []
        if self.use_real:
            uqil_lst.append(self.get_uqiltt(ind=ind))
        if self.use_rand:
            uqil_lst.append(
                self.get_uqiltt_neg_rand(ind=ind, n_neg=self.n_neg))
        uqil_cat = torch.cat(uqil_lst)
        return uqil_cat


class GenUqiiTuple(DataGenerator):
    """ return DataLoader(user_query[n, 2], nids[n, 2], target[n, 2])"""
    def __init__(self, data, ind=None,
                 use_rand=True, n_neg_rand=2, use_uq=True, n_neg_uq=1,
                 batch_size=256, label_col='click'):
        super(GenUqiiTuple, self).__init__(data, ind, label_col)
        self.name = 'GenUqiiTuple'
        self.use_rand = use_rand
        self.n_neg_rand = n_neg_rand
        self.use_uq = use_uq
        self.n_neg_uq = n_neg_uq
        self.batch_size = batch_size
        assert any([use_rand, use_uq])

    def __call__(self, *args, ind=None, **kwargs):
        ind = self.ind if ind is None else ind
        uqii_lst = []
        if self.use_rand:
            uqii_lst.append(
                self.get_uqiitt_rand(ind=ind, n_neg=self.n_neg_rand)
            )
        if self.use_uq:
            try:
                uqii_lst.append(
                    self.get_uqiitt_on_uq(ind=ind, n_neg=self.n_neg_uq)
                )
            except:
                print('no positive sample')
        uqii_cat = torch.cat(uqii_lst)
        uq_pair = uqii_cat[:, :2]
        nids = uqii_cat[:, 2:4]
        target = torch.cat((
            (nids[:, 0] == nids[:, 0]).int().view(-1, 1),
            (nids[:, 0] == nids[:, 1]).int().view(-1, 1),
        ), dim=1).float()
        uqii_zip = TensorDataset(uq_pair, nids, target)
        # uqii_zip = [(uq, nid, l) for uq, nid, l in zip(uq_pair, nids, target)]
        uqil_iter = DataLoader(uqii_zip, batch_size=self.batch_size,
                               pin_memory=True, shuffle=True)
        return uqil_iter


class GetUqitlTuple(GetUqilTuple):
    """
    return DataLoader(uqt[n, 3], nids[n, 1], target[n, 1])
    """
    def __init__(self, data, ind=None,
                 use_real=True, use_rand=False, n_neg=2, batch_size=256,
                 max_clk_seq=20, label_col='click'):
        super(GetUqitlTuple, self).__init__(
            data, ind, use_real=use_real, use_rand=use_rand, n_neg=n_neg,
            batch_size=batch_size, label_col=label_col
        )
        self.name = 'GetUqitlTuple'
        self.max_clk_seq = max_clk_seq

        self.iseq_tr, self.tseq_tr = self.create_u_seqs(data.ind_tr)
        self.test_mode = None
        self.iseq_te, self.tseq_te = None, None

    def __call__(self, *args, ind=None, **kwargs):
        ind = self.ind if ind is None else ind
        uqiltt = self.collect_data(ind)
        # time = uqiltt[:, 4:6].max(dim=1)[0]
        time = uqiltt[:, 4]
        uqt = torch.cat((uqiltt[:, :2], time.view(-1, 1)), dim=1)
        nids = uqiltt[:, 2].view(-1, 1)
        target = uqiltt[:, 3].float().view(-1, 1)
        print('for each current user generating pre-clicked sequence...')
        pre_clicks = self.get_u_iseq(uqt, self.iseq_tr, self.tseq_tr)
        uqitl_zip = [
            (uqt_, ck, i, l) for uqt_, ck, i, l in zip(uqt, pre_clicks, nids, target)
        ]
        uqitl_iter = DataLoader(uqitl_zip, batch_size=self.batch_size, shuffle=True)
        return uqitl_iter

    def create_u_seqs(self, ind):
        print('create user ck item sequence...')
        iseq_dict = dict.fromkeys(range(self.n_u), np.zeros(1))
        tseq_dict = dict.fromkeys(range(self.n_u), np.zeros(1))
        uids, iseqs_u, tseqs_u = self.get_cki(ind=ind, by='uid')
        for uid, iseq in zip(uids, iseqs_u):
            iseq_dict[uid] = np.concatenate([iseq_dict[uid], iseq])
        for uid, tseq in zip(uids, tseqs_u):
            tseq_dict[uid] = np.concatenate([tseq_dict[uid], tseq])
        return iseq_dict, tseq_dict

    def get_u_iseq(self, uqt, iseq_dict, tseq_dict):
        """
        create user's ck item sequence
        - for each user-time, generate last <max_len> item seq before t
        - if the uq sequence is shorter than max_len, append nid to the last
        :param uqt: T[n, 3+]
        :param iseq_dict: {u: iseq}
        :param tseq_dict: {u: tseq}
        :return iseq_pad: T[n, pad_seq_len]
        """
        len_pre_max = self.max_clk_seq - 1
        iseq_collect = []
        for uqt_i in uqt:
            u, t = uqt_i[0].item(), uqt_i[2].item()
            iseq_u, tseq_u = iseq_dict[u], tseq_dict[u]  # L[array[seq]], L[array[seq]]
            idx_time = np.where(tseq_u < t)[0]
            idx_start = len(idx_time) - len_pre_max if len(idx_time) > len_pre_max else 0
            idx_end = len(idx_time)
            iseq_collect.append(torch.tensor(iseq_u[idx_start:idx_end, ]))
        iseq_pad = pad_sequence(iseq_collect, batch_first=True)
        return iseq_pad.long()

    @staticmethod
    def expand_uqt_nids(uqt, nids):
        """
        expand nids with shape [n, *] to [n_i, 1]
        :param uqt: T[n, 3+], first three cols should be [uid, qid, pv_time]
        :param nids: T[n, *]
        :return: uqt_expand T[n_item, 3+], nids_expand T[n_item, 1]
        """
        n, n_pad = nids.shape
        uqt_expand = []
        nids_expand = []
        for i in range(n_pad):
            uqt_expand.append(uqt)
            nids_expand.append(nids[:, i])
        uqt_expand = torch.cat(uqt_expand)
        nids_expand = torch.cat(nids_expand).view(-1, 1)
        return uqt_expand, nids_expand

    def get_data_test(self, ind, uqt, nids, batch_size):
        if self.iseq_te is None:
            self.iseq_te, self.tseq_te = self.create_u_seqs(ind)
        pre_clicks = self.get_u_iseq(uqt, self.iseq_te, self.tseq_te)
        data_zip = [(uqt_, ck, nid) for uqt_, ck, nid in zip(uqt, pre_clicks, nids)]
        data_iter = DataLoader(data_zip, batch_size=batch_size)
        return data_iter


class GetUqtiilTuple(GetUqitlTuple):
    def __init__(
            self, data, ind=None, use_real=True, use_rand=False,
            n_neg_uq=1, n_neg_rand=2, batch_size=256,
            max_clk_seq=20, label_col='click'):
        super(GetUqtiilTuple, self).__init__(
            data, ind, use_real=use_real, use_rand=use_rand,
            batch_size=batch_size, max_clk_seq=max_clk_seq, label_col=label_col
        )
        self.n_neg_uq = n_neg_uq
        self.n_neg_rand = n_neg_rand

    def __call__(self, *args, ind=None, **kwargs):
        ind = self.ind if ind is None else ind
        uqiitt = self.collect_data(ind)
        time = uqiitt[:, 4]
        uqt = torch.cat((uqiitt[:, :2], time.view(-1, 1)), dim=1)
        nids = uqiitt[:, 2:4]
        target = torch.cat((
            (nids[:, 0] == nids[:, 0]).float().view(-1, 1),
            (nids[:, 0] == nids[:, 1]).float().view(-1, 1),
        ), dim=1).float()
        print('for each current user generating pre-clicked sequence...')
        pre_clicks = self.get_u_iseq(uqt, self.iseq_tr, self.tseq_tr)
        data_zip = [
            (uqt, ck, nid, l) for uqt, ck, nid, l in zip(uqt, pre_clicks, nids, target)]
        data_iter = DataLoader(
            data_zip, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        return data_iter

    def collect_data(self, ind):
        data_lst = []
        if self.use_real:
            try:
                data_lst.append(
                    self.get_uqiitt_on_uq(ind=ind, n_neg=self.n_neg_uq)
                )
            except:
                print('no positive sample')
        if self.use_rand:
            data_lst.append(
                self.get_uqiitt_rand(ind=ind, n_neg=self.n_neg_rand)
            )
        data_cat = torch.cat(data_lst)
        return data_cat


class GetAmazonUqiil(GetUqitlTuple):
    def __init__(
            self, data, ind=None, use_real=True, use_rand=False,
            n_neg_uq=1, n_neg_rand=2, batch_size=256,
            max_clk_seq=20, label_col='click'):
        super(GetAmazonUqiil, self).__init__(
            data, ind, use_real=use_real, use_rand=use_rand,
            batch_size=batch_size, max_clk_seq=max_clk_seq, label_col=label_col
        )
        self.n_neg_uq = n_neg_uq
        self.n_neg_rand = n_neg_rand
        self.tokens_r, self.lens_r = data.tokens_r, data.lens_r

    def __call__(self, *args, ind=None, **kwargs):
        ind = self.ind if ind is None else ind
        uqiitt = self.collect_data(ind)
        time = uqiitt[:, 4]
        uqt = torch.cat((uqiitt[:, :2], time.view(-1, 1)), dim=1)
        nids = uqiitt[:, 2:4]
        idx_tensor = torch.cat([torch.tensor(ind)] * self.n_neg_rand, dim=0).view(-1, 1)
        target = torch.cat((
            (nids[:, 0] == nids[:, 0]).float().view(-1, 1),
            (nids[:, 0] == nids[:, 1]).float().view(-1, 1),
        ), dim=1).float()
        print('for each current user generating pre-clicked sequence...')
        pre_clicks = self.get_u_iseq(uqt, self.iseq_tr, self.tseq_tr)
        data_zip = TensorDataset(uqt, pre_clicks, nids, target, idx_tensor)
        # data_zip = [
        #     (uqt, ck, nid, l, idx)
        #     for uqt, ck, nid, l, idx
        #     in zip(uqt, pre_clicks, nids, target, idx_tensor)
        # ]
        data_iter = DataLoader(
            data_zip, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        return data_iter

    def collect_data(self, ind):
        uqiitt = self.get_uqiitt_rand(ind=ind, n_neg=self.n_neg_rand)
        return uqiitt


class AmazonUqii(DataGenerator):
    """ return DataLoader(user_query[n, 2], nids[n, 2], target[n, 2])"""
    def __init__(self, data, ind=None,
                 use_rand=True, n_neg_rand=2, use_uq=True, n_neg_uq=1,
                 batch_size=256, label_col='click'):
        super(AmazonUqii, self).__init__(data, ind, label_col)
        self.name = 'AmazonUqii'
        self.use_rand = use_rand
        self.n_neg_rand = n_neg_rand
        self.use_uq = use_uq
        self.n_neg_uq = n_neg_uq
        self.batch_size = batch_size
        assert any([use_rand, use_uq])
        self.tokens_r, self.lens_r = data.tokens_r, data.lens_r

    def __call__(self, *args, ind=None, **kwargs):
        ind = self.ind if ind is None else ind
        uqii = self.get_uqiitt_rand(ind=ind, n_neg=self.n_neg_rand)

        uq_pair = uqii[:, :2]
        nids = uqii[:, 2:4]
        idx_tensor = torch.cat([torch.tensor(ind)] * self.n_neg_rand, dim=0).view(-1, 1)
        target = torch.cat((
            (nids[:, 0] == nids[:, 0]).int().view(-1, 1),
            (nids[:, 0] == nids[:, 1]).int().view(-1, 1),
        ), dim=1).float()
        uqii_zip = TensorDataset(uq_pair, nids, target, idx_tensor)
        uqil_iter = DataLoader(uqii_zip, batch_size=self.batch_size,
                               pin_memory=True, shuffle=True)
        return uqil_iter
