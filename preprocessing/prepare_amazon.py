import gzip
import json
import random
from pathlib import Path
from tqdm import tqdm

import pandas as pd

from preprocessing.prepare_data import PrepareData
from preprocessing import text_process


class PrepareAmazon(PrepareData):
    def __init__(self, data_path, name='amazon'):
        super(PrepareAmazon, self).__init__(data_path, name)
        self.min_word_r, self.min_word_q = 5, 0

    def load_data(self):
        output_path = Path(self.data_path, '../', self.name)
        output_path.mkdir(parents=True, exist_ok=True)

        print('> load data {}'.format(self.name))
        print('> load reviews')
        # review_path = Path(self.data_path, 'reviews_{}_5.json.gz'.format(self.name))
        review_path = Path(self.data_path, 'reviews_{}.json.gz'.format(self.name))
        i = 0
        df = {}
        g = gzip.open(review_path, 'rb')
        for line in tqdm(g):
            try:
                df[i] = json.loads(line)
                i += 1
            except KeyError:
                print('key missing once')
        review_df = pd.DataFrame.from_dict(df, orient='index')
        review_df = review_df[[
            'reviewerID', 'asin', 'reviewText',
            'unixReviewTime', 'reviewTime', 'overall'
        ]]

        print('> load meta information')
        meta_path = Path(self.data_path, 'meta_{}.json.gz'.format(self.name))
        with gzip.open(meta_path, 'rb') as g:
            i = 0
            df = {}
            categories = {}
            for line in tqdm(g):
                # df[i] = json.loads(line)
                df[i] = eval(line)
                try:
                    df[i]['category'] = df[i]['categories']
                except:
                    df[i]['categories'] = df[i]['category']
                try:
                    categories[df[i]['asin']] = df[i]['categories']
                except KeyError:
                    categories[df[i]['asin']] = self.name.replace('_', ' ')
                i += 1
            item_df = pd.DataFrame.from_dict(df, orient='index')
            titles, side_info = [], []
            # magazine: brand, publisher, description, //(also view & buy)
            # software: brand, rank, title, description//also_viewed
            # kindle: brand, rank, details.Publisher:, details.Language: (also_view, also_buy)

            for i in tqdm(range(item_df.shape[0])):
                t = item_df['title'][i]
                text = text_process._remove_char(t)
                text = ' '.join(text).lower()
                titles.append(text)
                try:
                    brand = 'its brand is {}, '.format(item_df['brand'][i])
                except:
                    brand = ''
                try:
                    publisher = 'its publisher is {}, '.format(item_df['details'][i]['Publisher:'])
                except:
                    publisher = ''
                try:
                    descrip = '{}. '.format(item_df['description'][i])
                except:
                    descrip = ''
                # try:
                #     lang = 'its language is {}, '.format(item_df['details'][i]['Language:'])
                # except:
                #     lang = ''
                try:
                    title = '{}, '.format(item_df['title'])
                except:
                    title = ''
                # try:
                #     rank = 'it ranks {}, '.format(item_df['rank'])
                # except:
                #     rank = ''

                # magazine: brand, publisher, description
                # info = '{}'.format(publisher)  # suffix1
                # info = '{}{}'.format(brand, publisher)  # suffix
                # info = '{}{}{}'.format(brand, publisher, descrip)  # suffix_descrip

                # software: brand, rank, title, description
                # info = '{}{}'.format(title)
                # info = '{}{}{}'.format(title, brand, rank)
                # info = '{}{}{}{}'.format(title, brand, rank, description)

                # kindle
                # info = '{}{}{}'.format(brand, descrip, lang)
                info = text_process._remove_char(info)
                info = ' '.join(info).lower()
                side_info.append(info)

            # for t in tqdm(item_df['title']):
            #     text = text_process._remove_char(t)
            #     text = ' '.join(text).lower()
            #     titles.append(text)
            item_df['title'] = titles
            item_df['side_info'] = side_info

            categories = []
            for category in tqdm(item_df['categories']):
                cate = [c for c in category]
                # cate = [c for sub_c in category for c in sub_c]
                if len(cate) == 0:
                    cate = [self.name.replace('_', ' ')]
                # else:
                #     if self.name in cate:
                #         cate.remove(self.name.replace('_', ' '))
                cate = [text_process._remove_char(t) for t in cate]
                cate = [' '.join(t).lower() for t in cate]
                categories.append(cate)
            item_df['categories'] = categories

            description = []
            for t in tqdm(item_df['description']):
                text = text_process._remove_char(t)
                text = ' '.join(text).lower()
                if text == 'nan':
                    text = self.name.replace('_', ' ')
                description.append(text)
            item_df['description'] = description

            # make item unique
            item_df_group = item_df.groupby(['asin'])
            item_lst = []
            for item, df in item_df_group:
                tmp = df.iloc[0, :]
                item_lst.append(dict(
                    asin=item, title=tmp['title'],
                    categories=tmp['categories'],
                    description=tmp['description'],
                    side_info=tmp['side_info']
                ))
            item_df = pd.DataFrame.from_dict(item_lst)
            item_df = item_df[['asin', 'title', 'categories', 'description', 'side_info']]
            self.raw['item'] = item_df

        # filter reviews outside meta
        set_item = self.raw['item']['asin'].unique()
        review_df = review_df[review_df['asin'].isin(set_item)]

        print('process queries')
        queries = []
        categories = item_df[['asin', 'categories']].set_index('asin')
        for i in tqdm(range(len(review_df))):
            asin = review_df['asin'].iloc[i]
            category = categories.loc[asin, 'categories']

            # process queries
            qs = list(map(text_process._remove_dup, map(text_process._remove_char, category)))
            try:
                q = random.choice(qs)
            except:
                q = self.name.replace('_', ' ')

            # process reviews
            queries.append(' '.join(q).lower())

        review_df['query_'] = queries
        self.raw['reviews'] = review_df

        # trans_path_outp = Path(output_path, 'tmp_trans_{}.csv'.format(self.name))
        # item_path_outp = Path(output_path, 'tmp_item_{}.csv'.format(self.name))
        # print('write', trans_path_outp, item_path_outp)
        # self.raw['reviews'].to_csv(trans_path_outp, index=False)
        # self.raw['item'].to_csv(item_path_outp, index=False)

    def preprocess_review(self):
        print('process reviews')
        reviews = []
        for i in tqdm(range(self.trans.shape[0])):
            review = self.trans['reviewText'].iloc[i]
            review = text_process._remove_char(review)
            reviews.append(' '.join(review).lower())
        self.trans['reviewText'] = reviews

    def prepare_data(self):
        print('> pre-process data {}'.format(self.name))
        # data, uid, qid, nid should be 0-based, text split by spaces
        # label cols dtypes are int, values no larger than 0 will be considered negative
        trans = self.raw['reviews']
        set_item = trans['asin'].unique()
        set_query = trans['query_'].unique()
        set_user = trans['reviewerID'].unique()

        # self.item = None  # pd.df [nid, item, {features}, ...]
        item = self.raw['item'][self.raw['item']['asin'].isin(set_item)]
        item['categories'] = ['@'.join(c) for c in item['categories']]
        item = item.groupby(['asin', 'title', 'categories', 'description', 'side_info']).size()
        # item = item.reset_index()[['asin', 'title', 'categories', 'description']]
        item = item.reset_index()[['asin', 'title', 'categories', 'description', 'side_info']]
        item['categories'] = [c.split('@') for c in item['categories']]

        item['nid'] = list(range(len(item)))
        self.item = item[['nid', 'title', 'asin', 'categories', 'description', 'side_info']]
        self.item.columns = ['nid', 'item', 'asin', 'cate_id', 'description', 'side_info']

        # self.user = None  # pd.df [uid, {features}, ...]
        self.user = pd.DataFrame(dict(
            uid=range(len(set_user)), reviewerID=set_user
        ))

        # self.query = None  # pd.df [qid, query, {features}, ...]
        self.query = pd.DataFrame(dict(
            qid=range(len(set_query)), query_=set_query
        ))

        # self.trans = None  # pd.df [uid, qid, nid, {label_cols}, {time}, ...]
        trans = trans.merge(self.item, how='left', on='asin')
        trans = trans.merge(self.user, how='left', on='reviewerID')
        trans = trans.merge(self.query, how='left', on='query_')
        trans['reviewText'] = trans['side_info'] + trans['reviewText']
        self.query.columns = ['qid', 'query']
        trans['click'] = 1
        self.trans = trans
        self.build_session(duration=3600*24*7)
        self.build_uq_time()

        self.trans = self.trans[[
            'uid', 'qid', 'nid', 'click',
            'unixReviewTime', 'unixReviewTime', 'unixReviewTime',
            'uq_time',
            # 'reviewTime',
            # 'overall',
            'reviewText', 'cate_id', 'side_info'
        ]]
        self.trans.columns = [
            'uid', 'qid', 'nid', 'click',
            'pv_time', 'ck_time', 'py_time',
            'uq_time',
            # 'reviewTime', 'overall',
            'reviewText', 'cate_id', 'side_info'
        ]
        self.trans['click_pos'] = 1

    def build_session(self, duration=1800):
        # duration = 3600 * 24 * 30
        print('build session for {} with {} s'.format(self.name, duration))
        self.trans = self.trans.sort_values(['uid', 'unixReviewTime'])
        self.trans['time_diff'] = self.trans['unixReviewTime'].diff()
        self.trans['flag_duration'] = self.trans['time_diff'] >= duration
        self.trans['flag_user'] = self.trans['uid'].diff() == 1
        flag = self.trans['flag_duration'] | self.trans['flag_user']
        flag.iloc[0] = True
        flag = flag.apply(int)
        self.trans['sid'] = flag.cumsum()

    def build_uq_time(self):
        trans_group = self.trans.groupby(['sid', 'qid'])
        uq_time = trans_group['unixReviewTime'].min().reset_index(name='uq_time')
        self.trans = self.trans.merge(uq_time, how='left', on=['sid', 'qid'])

    def filter_u_min_q(self, min_q=3):
        """ u_min_q@<min_q>
        filter out users with no more than <min_q> queries
        return if all the users have at least <min_q> queries
        """
        min_q = int(min_q)
        uq = self.trans.groupby(['uid', 'qid']).size().reset_index(name='freq')
        u_count_q = uq.groupby('uid').size().reset_index(name='count_q')
        u_keep = u_count_q[u_count_q['count_q'] >= min_q]['uid']
        flag = u_keep.shape[0] == self.user.shape[0]
        if not flag:
            self.trans = self.trans[self.trans['uid'].isin(u_keep)]
            self._remap_data()
            self.log_data()
        return flag

    def filter_u_core(self, core=5):
        """ u_core@<core>
        filter out users with no more than <min_q> queries
        return if all the users have at least <min_q> queries
        """
        core = int(core)
        user_freq = self.trans.groupby(['uid']).size().reset_index(name='freq')
        u_keep = user_freq[user_freq['freq'] >= core]
        flag = u_keep.shape[0] == self.user.shape[0]
        if not flag:
            self.trans = self.trans[self.trans['uid'].isin(u_keep)]
            self._remap_data()
            self.log_data()
        return flag

    def filter_i_core(self, core=5):
        """ i_core@<core>
        filter out users with no more than <min_q> queries
        return if all the users have at least <min_q> queries
        """
        core = int(core)
        item_freq = self.trans.groupby(['nid']).size().reset_index(name='freq')
        i_keep = item_freq[item_freq['freq'] >= core]
        flag = i_keep.shape[0] == self.item.shape[0]
        if not flag:
            self.trans = self.trans[self.trans['nid'].isin(i_keep)]
            self._remap_data()
            self.log_data()
        return flag

    def log_data(self):
        self.log['name'] = self.name
        self.log['n'] = self.trans.shape[0]
        self.log['n_u'] = self.user.shape[0]
        self.log['n_q'] = self.query.shape[0]
        self.log['n_i'] = self.item.shape[0]
        self.log['filters'] = self.filters
        print(self.log)


def prepare_data(prefix_inp, prefix_outp, name, filters):
    """filters refer to self.filter_<*> keys"""
    data = PrepareAmazon(prefix_inp, name=name)
    data.prepare_data()
    data.filter_data(filters)
    data.preprocess_review()
    data.output_data(prefix_outp)


if __name__ == '__main__':

    filters = ['u_min_q@4']
    prefix_input = '../data/amazon/raw/2018'

    name = 'Magazine_Subscriptions'
    prefix_output = '../data/amazon/mm/Magazine_Subscriptions'

    name = 'Software'
    prefix_output = '../data/amazon/mm/Software'

    prepare_data(prefix_input, prefix_output, name, filters)
