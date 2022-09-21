from pathlib import Path
import pandas as pd
import torch

from persearch.utils import flatten
import re


def _remove_char(value):
    value = re.sub(
        '[\[\],!.;#$^*\_——<>/=%&?@"&\'-:]', ' ', str(value))
    l_temp = [i for i in value.split()]
    text = ' '.join(l_temp).lower()
    return text


class DataLoader(object):
    """
    1. load: load well-cooked data in the beginning
    2. tokenize: get token_q/i, lens_q/i
    3. pick: subset and split data, charge data mask self.ind_tr/val/te
    """
    def __init__(self, data_path, name, preprocess=False):

        self.name = name
        self.data_path = data_path
        self.preprocess = preprocess

        # load
        self.trans = None  # pd.df [uid, qid, nid, {label_cols}, ...]
        self.user = None  # pd.df [uid, {features}, ...]
        self.query = None  # pd.df [qid, query, {features}, ...]
        self.item = None  # pd.df [nid, item, {features}, ...]
        self.load_data()
        self.addition_process()

        # tokenize text, tokens_q/i, lens_q/i
        self.corpus = self.build_corpus()
        self.tokenizer = None
        self.n_v = None
        self.tokens_q, self.lens_q = None, None  # tensor[n_q, max_len], tensor
        self.tokens_i, self.lens_i = None, None
        self.tokens_r, self.lens_r = None, None
        self.q90 = 256

        # subset and split data
        self.picker = None
        self.ind_sel = None
        self.ind_tr, self.ind_val, self.ind_te = None, None, None

        # others, device, log, etc.
        self.device = torch.device('cpu')

    def load_data(self):
        print('> load data {} from {}'.format(self.name, self.data_path))
        prefix = Path(self.data_path, self.name)
        self.trans = pd.read_csv('{}_trans.csv'.format(prefix), sep=';')
        self.user = pd.read_csv('{}_user.csv'.format(prefix), sep=';')
        self.query = pd.read_csv('{}_query.csv'.format(prefix), sep=';')
        self.item = pd.read_csv('{}_item.csv'.format(prefix), sep=';')
        print()

    def addition_process(self):
        self.trans = self.trans.fillna(-1)
        try:
            self.trans = self.trans.astype(dict(
                pv_time='int64', ck_time='int64', py_time='int64'
            ))
        except:
            print('no time information')
        if 'sid' not in self.trans.columns:
            self.build_session()
        if not self.preprocess:
            if 'side_info' in self.item.columns:
                self.item['side_info'] = [eval(r) for r in self.item['side_info']]
                trans = self.trans.merge(self.item, how='left', on='nid')
                trans['side_info'] = trans['side_info'].apply(lambda x: ' '.join(x))
                self.trans['reviewText'] = trans['side_info'] + ' ' + self.trans['reviewText']
            # self.trans['reviewText'] = self.trans['reviewText'].apply(_remove_char)

    def build_corpus(self):
        """return [docs], collect all possible text"""
        print('> build corpus for {}'.format(self.name))
        corpus = self.query['query'].tolist() + self.item['item'].tolist()
        try:
            corpus += self.trans['reviewText'].tolist()
        except:
            print('no review')
        # if 'side_info' in self.item.columns:
        #     corpus += flatten(self.item['side_info'])
        return corpus

    def tokenize(self, tokenizer):
        """update tokens_q/i, lens_q/i using tokenizer"""
        self.tokenizer = tokenizer
        self.n_v = len(tokenizer.dictionary)
        self.tokens_q, self.lens_q = self.tokenizer.tokenize(self.query['query'].tolist())
        self.tokens_i, self.lens_i = self.tokenizer.tokenize(self.item['item'].tolist())
        try:
            self.tokens_r, self.lens_r = self.tokenizer.tokenize(
                self.trans['reviewText'].tolist(), if_pad=False)
            self.tokens_r = self.tokens_r[1:]
            self.lens_r = self.lens_r[1:]
            lens_r = pd.Series(self.lens_r)
            if self.name == 'CDs_and_Vinyl':
                self.q90 = int(lens_r.quantile(0.75))
            else:
                self.q90 = int(lens_r.quantile(0.9))
            self.tokens_r = [tk_r[:self.q90] for tk_r in self.tokens_r]
            self.lens_r[self.lens_r > self.q90] = self.q90
            # self.q90 = 256
        except:
            print('no review')
        self.device = torch.device('cpu')

    def pick(self, picker=None):
        """subset and split data, charge ind_tr/val/te"""
        if picker is not None:
            assert len(picker.p) == 3
            self.picker = picker
        self.ind_tr, self.ind_val, self.ind_te = self.picker.pick(self.trans)
        self.ind_sel = self.ind_tr + self.ind_val + self.ind_te

    def set_picker(self, picker):
        assert len(picker.p) == 3
        self.picker = picker

    def to(self, device: torch.device):
        print('> mv data {} to {}'.format(self.name, device))
        self.device = device
        self.tokens_q = self.tokens_q.to(device)
        self.lens_q = self.lens_q.to(device)
        self.tokens_i = self.tokens_i.to(device)
        self.lens_i = self.lens_i.to(device)

    def build_session(self, duration=604800):
        # duration = 3600 * 24 * 7 (one week) (use default for amazon)
        # 86400, 604800, 2592000, 7776000, 30758400
        print('build session for {} with {} s'.format(self.name, duration))
        trans_tmp = self.trans.sort_values(['uid', 'py_time'])
        trans_tmp['time_diff'] = trans_tmp['py_time'].diff()
        trans_tmp['flag_duration'] = trans_tmp['time_diff'] >= duration
        trans_tmp['flag_user'] = trans_tmp['uid'].diff() == 1
        flag = trans_tmp['flag_duration'] | trans_tmp['flag_user']
        flag.iloc[0] = True
        flag = flag.apply(int)
        trans_tmp['sid'] = flag.cumsum()
        trans_tmp = trans_tmp[['uid', 'qid', 'nid', 'pv_time', 'sid']]
        self.trans = self.trans.merge(
            trans_tmp, how='left', on=['uid', 'qid', 'nid', 'pv_time'])
