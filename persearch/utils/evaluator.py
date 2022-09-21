import torch
import persearch.utils.eval as evl
from persearch.config import get_eval

from persearch.data.generator import DataGenerator
from persearch.utils import print_title
from persearch.utils.summary import Summary


class Evaluator(object):
    """
    - prepare test data from DataLoader based on eval_keys
        - build evaluation and charge self.tests
    - evaluate model and return [{metric: value}]
    :param data: DataLoader object
    :param eval_names: keys to get evals [dict(eval_name, eval_key, params, metrics)]
        - eval_name: str, indicating a unique evaluation
        - eval_key: str, refer to Evaluator.build_<eval_key>
        - params: dict, parameters of build_<eval_key>
        - metrics: [metric]
    - methods
        - _build_tests(): build all the tests by evals
        - _build_<>: return test, add to self.eval
            :return test: dict(predict_mode, data, truth)
                - predict_mode: a model should has predict_<predict_mode>
                - data: data required by model.predict_<predict_mode>
                - truth: truth for this evaluation
        - evaluate(model) -> rst: list
            - evaluate the model on all the test
            :return rst: dict(eval_name, {metric: value})
    """
    def __init__(self, data, eval_names, prefix='../logs', is_test=True):
        self.ind = data.ind_te if is_test is None else data.ind_val
        self.ind_tr = data.ind_tr
        self.ind_sel = data.ind_tr + self.ind
        self.prefix = prefix
        self.generator = DataGenerator(data)
        self.eval_names = eval_names
        self.evals = []
        self._build_tests()

    def _build_tests(self):
        """prepare data and get predict_mode for all the evals"""
        for eval_name in self.eval_names:
            print('> build {}'.format(eval_name))
            eval_ = get_eval(eval_name).copy()
            eval_key = eval_['eval_key']
            params = eval_['params']
            f_build = getattr(self, '_build_{}'.format(eval_key))
            eval_.update(f_build(**params))
            eval_['summary'] = Summary(
                name=eval_name,
                metrics=[k for k in eval_['metrics'].keys()]
            )
            self.evals.append(eval_)

    def evaluate(self, model, print_key=None):
        model.eval()
        rst = dict()
        print_key = model.name if print_key is None else print_key
        print_title('{} EVAL START'.format(print_key))
        for test in self.evals:
            eval_name = test['eval_name']
            print('{:<20}'.format(eval_name), end=' || ')
            predict_mode = test['predict_mode']
            data_test, target = test['data'], test['truth']
            pred = model.predict(data_test, predict_mode, self.ind_sel)
            # target = target.to(pred.device)
            pred = pred.to(target.device)
            metrics = test['metrics']
            rst[eval_name] = dict.fromkeys(metrics, 0)
            for metric_key, metric in metrics.items():
                f_eval = getattr(evl, metric['name'])
                val = f_eval(pred, target, **metric)
                val = val if isinstance(val, float) else int(val)
                rst[eval_name][metric_key] = val
            test['summary'].add(print_key, rst[eval_name])
            self.print_rst(rst[eval_name])
            del pred, target
        print_title('-' * 20)
        return rst

    def write(self, dir_outp=''):
        for test in self.evals:
            test['summary'].write(test['summary'].rst[-1], dir_outp=dir_outp)

    def print(self):
        for test in self.evals:
            print(test['summary'].rst[-1])

    @staticmethod
    def print_rst(rst):
        counter = 0
        n_metric = len(rst)
        for metric, val in rst.items():
            str_val = '{:.4f}'.format(val) if isinstance(val, float) else '{}'.format(val)
            str_print = '{:<8}: {}'.format(metric, str_val)
            print('{:<20}'.format(str_print), end=' | ')
            counter += 1
            if (counter % 4 == 0) & (counter != n_metric):
                print('\n{:<20}'.format(''), end=' || ')
        print()

    def _build_uq_topk_mix(self, topk=100):
        """
        for each uq pair, mix the negative samples and pad it to topk items
        """
        uqt_test, pad_truth, topk_mix = self.generator.get_topk_mix_uq(
            ind=self.ind, topk=topk)
        data_test = uqt_test, topk_mix
        test = dict(
            predict_mode='topk_mix',
            data=data_test,
            truth=pad_truth,
        )
        return test

    def _build_uqi_frequent_mix(self, topk=100, lim_freq=1):
        """
        for each uq pair, mix the negative samples and pad it to topk items
        """
        uqit_test, pad_truth, topk_mix = self.generator.get_topk_mix_uqi(
            ind=self.ind, topk=topk)
        data_tr = self.generator.trans.loc[self.ind_tr, :]
        freq_i_tr = data_tr.groupby('nid').size().reset_index(name='freq')
        freq_i = freq_i_tr[freq_i_tr['freq'] <= lim_freq]['nid']
        i_lst = uqit_test.iloc[:, 2]
        flag_zero = ~i_lst.isin(freq_i_tr['nid'])
        if lim_freq == 0:
            flags = flag_zero
        else:
            flag_lim = i_lst.isin(freq_i)
            flags = flag_zero | flag_lim
        flags = torch.tensor(flags.values)
        uqit_test = torch.tensor(uqit_test.values, dtype=torch.long)
        uqit_test, pad_truth = uqit_test[flags, :], pad_truth[flags, :]
        topk_mix = topk_mix[flags, :]
        data_test = uqit_test, topk_mix
        test = dict(
            predict_mode='topk_mix',
            data=data_test,
            truth=pad_truth,
        )
        return test

    # def _build_uqi_frequent_mix(self, topk=100, lim_freq=1):
    #     """
    #     for each uq pair, mix the negative samples and pad it to topk items
    #     """
    #     uqit_test, pad_truth, topk_mix = self.generator.get_topk_mix_uqi(
    #         ind=self.ind, topk=topk)
    #     i_lst = uqit_test.iloc[:, 2]
    #     data_tr = self.generator.trans.loc[self.ind_tr, :]
    #     freq_i_tr = data_tr.groupby('nid').size().reset_index(name='freq')
    #     if lim_freq == 0:
    #         flags = ~i_lst.isin(freq_i_tr['nid'])
    #     else:
    #         if lim_freq < 4:
    #             freq_sel = freq_i_tr[freq_i_tr['freq'] == lim_freq]['nid']
    #         else:
    #             freq_sel = freq_i_tr[freq_i_tr['freq'] >= lim_freq]['nid']
    #         flags = i_lst.isin(freq_sel)
    #     flags = torch.tensor(flags.values)
    #     uqit_test = torch.tensor(uqit_test.values, dtype=torch.long)
    #     uqit_test, pad_truth = uqit_test[flags, :], pad_truth[flags, :]
    #     topk_mix = topk_mix[flags, :]
    #     data_test = uqit_test, topk_mix
    #     test = dict(
    #         predict_mode='topk_mix',
    #         data=data_test,
    #         truth=pad_truth,
    #     )
    #     return test

    @staticmethod
    def _transform_uqiltt(uqiltt):
        uqtt = torch.cat([uqiltt[:, :2], uqiltt[:, 4:6]], dim=1)
        nid = uqiltt[:, 2].view(-1, 1)
        data_test = uqtt, nid
        label = uqiltt[:, 3]
        return data_test, label
