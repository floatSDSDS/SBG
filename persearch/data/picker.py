from tqdm import tqdm


class DataPicker(object):
    """ subset and split input transaction
        __init__(self, strategy: str, use_p: float=1.0, p: list=[0.6, 0.2, 0.2])
        params strategy: str  (u_last2_uq)
        params use_p: float in (0, 1]
        method
            pick_<>(trans)
                - input trans: pd.df [uid, qid, nid, label_col]
                - return: [[ind_i]]: list of index corresponding to trans

        * strategy
            - u_last2_uq:
                - subset p of users
                - last query as test, second last as valid
    """
    def __init__(self, strategy='trans', use_p=1.0, p=None):
        self.strategy = strategy
        self.use_p = use_p
        self.p = [0.6, 0.2, 0.2] if p is None else p
        self.pick = getattr(self, 'pick_{}'.format(self.strategy))

    def pick_u_last2_uq(self, trans):
        print('> subset {} of users, use last 2 uq for test and valid'.format(self.use_p))
        ind_subset = self._subset_u(trans, use_weight=False)
        trans_subset = trans.loc[ind_subset, :]
        return self._split_uq_last2_pos(trans_subset)

    def _subset_u(self, trans, use_weight=False):
        u_freq = trans.groupby('uid').size().reset_index(name='freq')
        u_subset = u_freq.sample(frac=self.use_p, weights='freq' if use_weight else None)
        uid_subset = u_subset['uid']
        trans_subset = trans[trans['uid'].isin(uid_subset)]
        return trans_subset.index.tolist()

    @staticmethod
    def _split_uq_last2_pos(trans_subset):

        trans_sort = trans_subset.sort_values(['uid', 'uq_time'])
        group_uq = trans_sort.groupby(['uid'])[['qid', 'uq_time']]
        uq_val, uq_te = [], []
        for u, df_u in tqdm(group_uq):
            df_pos = df_u.loc[df_u['click_pos'] > 0, ]
            uq_t = df_pos.groupby(['qid'])['uq_time']
            uq_t_sort = uq_t.min().sort_values().reset_index()
            uq_id_sort = uq_t_sort['qid'].tolist()
            try:
                uq_val.append((u, uq_id_sort[-2]))
                uq_te.append((u, uq_id_sort[-1]))
            except:
                continue

        trans_index_uq = trans_subset.copy()
        trans_index_uq['ind_ori'] = trans_index_uq.index
        trans_index_uq = trans_index_uq.set_index(['uid', 'qid'])

        ind_val = trans_index_uq.loc[uq_val]['ind_ori'].tolist()
        ind_te = trans_index_uq.loc[uq_te]['ind_ori'].tolist()
        ind_tr = trans_subset.drop(ind_val + ind_te).index.tolist()
        return [ind_tr, ind_val, ind_te]
