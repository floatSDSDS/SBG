from tqdm import tqdm

import torch
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import from_scipy_sparse_matrix

from persearch.data.generator import DataGenerator
from persearch.utils import flatten

"""Graph Generator
with given index of transaction, return the graph edge list
node index will be set with the order [item, other entities, pad node if needed]
    - item index are same with the ones in the transaction/data.item
__init__(data, ind, **kwargs)
    - data, DataLoader
    - ind, visible index in data.trans
    methods
        - build_edges() -> T[n_edge, 2]
"""


class GenGraphCk(DataGenerator):
    """add edge if two items are clicked successively"""
    def __init__(self, data, ind=None, label_col='click'):
        super(GenGraphCk, self).__init__(data, ind, label_col)
        ind_graph = data.ind_tr if ind is None else ind
        self.sids, self.iseqs, self.tseq = self.get_cki(ind=ind_graph)
        self.n_e = self.n_i

    def build_edges(self):
        edges = []
        bar = tqdm(enumerate(self.iseqs))
        for idx, seqs in bar:
            bar.set_description('Process session id {}'.format(idx + 1))
            for i in range(len(seqs) - 1):
                cur = seqs[i]
                nxt = seqs[i + 1]
                edges.append([cur, nxt])
        edges = torch.tensor(edges)
        return edges


class GenGraphTfidf(DataGenerator):
    """construct tfidf-cos similarity graph using all the title"""
    def __init__(self, data, ind=None, label_col='click', corpus='item'):
        super(GenGraphTfidf, self).__init__(data, ind, label_col)
        self.item = data.item
        self.ind_tr = data.ind_tr
        self.vectorizer = TfidfVectorizer()
        self.th_item_cos = 0.5
        self.corpus = corpus
        self.n_e = data.item.shape[0] + 1

    def build_edges(self):
        data = self.trans.loc[self.ind_tr, ['nid']]
        nid_freq = data.groupby(['nid']).size().reset_index(name='freq')
        items = self.item[['nid', self.corpus]]
        item_sel_freq = items.merge(nid_freq, how='left').fillna(0)
        item_sel_freq = item_sel_freq.set_index('nid')
        nid_train = data['nid']
        docs = item_sel_freq.loc[nid_train, self.corpus].tolist()
        ind_nan_tr = [i for i, v in enumerate(docs) if not isinstance(v, str)]
        print('process nan doc')
        for ind_d in ind_nan_tr:
            docs[ind_d] = 'null'
        print('fit item tfidf...')
        self.vectorizer.fit_transform(docs)
        docs_items = [''] + self.item[self.corpus].tolist()
        ind_nan_doc = [i for i, v in enumerate(docs_items) if not isinstance(v, str)]
        print('process nan doc')
        for ind_d in ind_nan_doc:
            docs_items[ind_d] = 'null'

        doc_term_sparse = self.vectorizer.transform(docs_items)
        print('calculate tfidf cos...')
        doc_tfidf_cos = cosine_similarity(doc_term_sparse, doc_term_sparse)
        doc_tfidf_cos[doc_tfidf_cos < self.th_item_cos] = 0
        sim_matrix = csr_matrix(doc_tfidf_cos)
        edge_index, edge_weight = from_scipy_sparse_matrix(sim_matrix)
        return edge_index.t()


class GenGraphQI(DataGenerator):
    """item-query click bipartite graph"""
    def __init__(self, data, gap_len=3, ind=None, label_col='click'):
        super(GenGraphQI, self).__init__(data, ind, label_col)
        self.n_query = data.trans['qid'].max()
        self.n_gap = gap_len
        self.n_e = self.n_i + self.n_query + self.n_gap

    def build_edges(self):
        data = self.trans.loc[self.ind, ['qid', 'nid', self.label_col]]
        data = data[data[self.label_col] > 0]
        data_group = data.groupby(['qid', 'nid']).size().reset_index(name='freq')
        data_group['qid'] += self.n_i - 1
        edges = data_group.loc[:, ['qid', 'nid']].values
        edges = torch.tensor(edges, dtype=torch.long)
        return edges


class GenGraphSI(DataGenerator):
    """item-session(id) bipartite graph"""
    def __init__(self, data, gap_len=0, ind=None, label_col='click'):
        super(GenGraphSI, self).__init__(data, ind, label_col)
        self.n_session = data.trans['sid'].max()
        self.n_gap = gap_len
        self.n_e = self.n_i + self.n_session + self.n_gap

    def build_edges(self):
        data = self.trans.loc[self.ind, ['sid', 'nid', self.label_col]]
        data = data[data[self.label_col] > 0]
        data_group = data.groupby(['sid', 'nid']).size().reset_index(name='freq')
        data_group['sid'] += self.n_i - 1
        edges = data_group.loc[:, ['sid', 'nid']].values
        edges = torch.tensor(edges, dtype=torch.long)
        return edges


class GenGraphUI(DataGenerator):
    """item-user click bipartite graph"""
    def __init__(self, data, gap_len=0, ind=None, label_col='click'):
        super(GenGraphUI, self).__init__(data, ind, label_col)
        self.n_user = data.trans['uid'].max()
        self.n_gap = gap_len
        self.n_e = self.n_i + self.n_user + self.n_gap

    def build_edges(self):
        data = self.trans.loc[self.ind, ['uid', 'nid', self.label_col]]
        data = data[data[self.label_col] > 0]
        data_group = data.groupby(['uid', 'nid']).size().reset_index(name='freq')
        data_group['uid'] += self.n_i - 1
        # data_group['uid'] += self.n_i
        edges = data_group.loc[:, ['uid', 'nid']].values
        edges = torch.tensor(edges, dtype=torch.long)
        return edges


class GenGraphCate(DataGenerator):
    """item-user click bipartite graph"""
    def __init__(self, data, gap_len=0, ind=None, label_col='click'):
        super(GenGraphCate, self).__init__(data, ind, label_col)

        if 'cate_id' not in data.item.columns:
            item_cate = data.trans.groupby(['nid', 'cate_id']).size().reset_index(name='freq')
            self.item = data.item.merge(item_cate, how='left', on='nid')
            self.item = self.item[['nid', 'cate_id']]
            self.item['cate_id'] = self.item['cate_id'].fillna(0)
            all_cate = [eval(str(c)) for c in self.item['cate_id'].unique()]
            self.unique_cate = list(set(all_cate))
            self.item['cate_id'] = [self.unique_cate.index(c) for c in self.item['cate_id']]
            self.is_ali = True
        else:
            self.item = data.item[['nid', 'cate_id']]
            all_cate = flatten([eval(c) for c in self.item['cate_id'].unique()])
            self.unique_cate = list(set(all_cate))
            self.item['cate_id'] = [[self.unique_cate.index(c) for c in eval(cates)] for cates in self.item['cate_id']]
            self.is_ali = False

        self.n_cate = len(self.unique_cate)
        self.n_gap = gap_len  # for padding or zero-item
        self.n_e = self.n_i + self.n_cate + self.n_gap

    def build_edges(self):
        edges = []
        for i in tqdm(range(self.item.shape[0])):
            nid, cates = self.item.iloc[i, 0], self.item.iloc[i, 1]
            if self.is_ali:
                edges.append([cates, nid])
            else:
                edges.append([[c, nid] for c in cates])

        if self.is_ali:
            pass
        else:
            edges = flatten(edges)
        edges = torch.tensor(edges, dtype=torch.long)
        edges[:, 0] += self.n_i - 1
        return edges


class GenGraphI(DataGenerator):
    """item-user click bipartite graph"""
    def __init__(self, data, ind=None, label_col='click'):
        super(GenGraphI, self).__init__(data, ind, label_col)
        self.n_e = self.n_i

    def build_edges(self):
        col = torch.tensor(range(self.n_e), dtype=torch.long).view(-1, 1)
        return torch.cat([col, col], dim=1)
