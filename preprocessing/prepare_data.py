from pathlib import Path
import pandas as pd
from persearch.utils import Config


class PrepareData(object):
    """
    load raw data, preprocess and output standard dataset
    preprocess:
        - transform to trans/user/query/item
        - filter and remap ids data
            - filter_<>
    """
    def __init__(self, data_path: str, name: str = 'data'):
        self.name = name  # str
        self.data_path = data_path  # str, path contains data
        self.filters = []  # list of filter keys

        self.raw = dict()  # stores named raw data

        # data, uid, qid, nid should be 0-based, text split by spaces
        # label cols dtypes are int, values no larger than 0 will be considered negative
        self.trans = None  # pd.df [uid, qid, nid, {label_cols}, {time}, ...]
        self.user = None  # pd.df [uid, {features}, ...]
        self.query = None  # pd.df [qid, query, {features}, ...]
        self.item = None  # pd.df [nid, item, {features}, ...]

        self.log = dict()
        self.load_data()

    def load_data(self):
        """load data to self.raw"""
        raise NotImplementedError

    def prepare_data(self):
        """turns raw data into self/trans/user/query/item"""

    def build_corpus(self) -> pd.DataFrame:
        """define corpus for w2v subset pd.df [space split text, frequency]"""

    def filter_data(self, filters):
        self.filters += filters
        self.filters = list(set(self.filters))
        print('> filter data with {}'.format(self.filters))
        flags = [False] * len(filters)
        while not all(flags):
            flags = []
            for key_filter in self.filters:
                if '@' in key_filter:
                    name_filter, param = key_filter.split('@')
                    filter_ = getattr(self, 'filter_{}'.format(name_filter))
                    flag = filter_(param)
                else:
                    filter_ = getattr(self, 'filter_{}'.format(key_filter))
                    flag = filter_()
                flags.append(flag)
                message = 'pass' if flag else 'work'
                print('{}: {}'.format(key_filter, message))

    def output_data(self, prefix: str = '.'):
        """output standard dataset,
        output self.trans/user/query/items
        version = <name>_{<ver>-<ver>}
        <version>_*.csv, sep=';'
        """
        output_path = Path(prefix)
        print('output dataset {} to {}'.format(self.name, output_path))
        output_path.mkdir(parents=True, exist_ok=True)
        self.trans.to_csv('{}/{}_trans.csv'.format(prefix, self.name), sep=';', index=False)
        self.user.to_csv('{}/{}_user.csv'.format(prefix, self.name), sep=';', index=False)
        self.query.to_csv('{}/{}_query.csv'.format(prefix, self.name), sep=';', index=False)
        self.item.to_csv('{}/{}_item.csv'.format(prefix, self.name), sep=';', index=False)
        output_log = Config(self.log)
        output_log.save(prefix, filename='datainfo_{}.json'.format(self.name))

    def _remap_data(self):
        """remap ids in data after filtering"""
        uid_map = self._remap_col('uid')
        self.trans['uid'] = self.trans['uid'].map(uid_map)
        self.user = self.user[self.user['uid'].isin(uid_map.keys())]
        self.user['uid'] = self.user['uid'].map(uid_map)
        self.user = self.user.sort_values('uid')

        qid_map = self._remap_col('qid')
        self.trans['qid'] = self.trans['qid'].map(qid_map)
        self.query = self.query[self.query['qid'].isin(qid_map.keys())]
        self.query['qid'] = self.query['qid'].map(qid_map)
        self.query = self.query.sort_values('qid')

        nid_map = self._remap_col('nid')
        self.trans['nid'] = self.trans['nid'].map(nid_map)
        self.item = self.item[self.item['nid'].isin(nid_map.keys())]
        self.item['nid'] = self.item['nid'].map(nid_map)
        self.item = self.item.sort_values('nid')

    def _remap_col(self, col: str) -> dict:
        """return 0-based remap ids for col with filtered self.trans"""
        ids_new = self.trans[col].unique()
        return {id_old: id_new + 1 for id_new, id_old in enumerate(ids_new)}
