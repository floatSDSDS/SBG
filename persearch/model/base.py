import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


import persearch.gens as gens
from persearch.utils.eval import auc, f_score
from persearch.utils import smart_sort, dist_cos, print_title, timing
from persearch.config import get_cfg_gen
from persearch.model.layer.layer import SelfAttEncoder


class Base(nn.Module):
    """ wrap model pipeline
    methods:
        - __init__(data:DataLoader, cfg_model:dict)
            - training data generator
            - model parameters
        - forward(ctx: T[n, *], nids: T[n, pad_len], *args) -> scores:T[n, pad_len]
        - do_train(ctx: T[n, *], nids: T[n, pad_len], *args)
            -> loss: T, scores: T[n, pad_len], pred: [n, pad_len],
            - feed in batch data, return loss, score, and prediction
        - fit(data: DataLoader)
            - generate training data
            - training and update params
        - _predict_*(test_data, ind_sel) ->
            - return prediction
        - move_to(device: torch.device)
            - move data such as tokenized item title to the specific device
    """
    def __init__(self, data, cfg_model):
        super(Base, self).__init__()

        self.name = cfg_model['name']
        self.data = data

        self.batch_size = cfg_model['batch_size']
        self.lr = cfg_model['lr']
        self.i_epoch = 0
        self.flag_train = True  # whether stop training
        self.flag_epoch = True  # if need initialization for each epoch

        key_generator = cfg_model['generator']
        cfg_gen = get_cfg_gen(key_generator)
        self.generator = getattr(gens, cfg_gen['gen'])(
            data, ind=data.ind_tr, batch_size=self.batch_size, **cfg_gen['param'])

        self.n_v = data.n_v
        self.d = cfg_model['d']

        # get pre-trained emb
        self.emb_v = nn.Embedding(self.n_v, self.d, padding_idx=0)
        self.tokens_q, self.lens_q = data.tokens_q, data.lens_q
        self.tokens_i, self.lens_i = data.tokens_i, data.lens_i
        self.tokens_r, self.lens_r = data.tokens_r, data.lens_r
        self.encoder = SelfAttEncoder(self.d, int(self.d/2))
        self.f_sim = dist_cos

        self.device = torch.device('cpu')
        print_title('BUILD {}'.format(self.name))

    def forward(self, *args) -> torch.FloatTensor:
        """ update loss, return score
        :return scores [n, pad_len]
        """

    def do_train(self, *args):
        """
        1. parse data_batch, call self.forward
        2. call loss, do prediction
        3. return loss, scores, and pred
        calculate and charge loss on data_train, return scores
        :param ctx: T[n, *], train unit generated by self.generator, such as uq
        :param nids: T[n, pad_len], item ids generated by self.generator
        :param target: T[n, pad_len], target need by self.f_loss
        :return loss: loss calculated by self.f_loss
        :return scores: [n, pad_len], returned by self(uq, nids)
        :return pred: [n, pad_len], pred=self.score_ctr_pred(scores)
        """

    def fit(self, step=10):
        print('> fit {}'.format(self.name))
        self.train()
        self.flag_train = True
        data_tr = self.generator(ind=self.data.ind_tr)
        optimizer = optim.Adam(
            [param for param in self.parameters() if param.requires_grad is True],
            self.lr, weight_decay=1e-5)
        while self.flag_train:
            # enter a epoch
            self.flag_epoch = True
            loss_avg = 0.0
            f1_avg = 0.0
            auc_avg = 0.0
            time_start = time.time()
            i_batch = 0
            for i_batch, data_batch in enumerate(data_tr):
                data_batch = self.batch_to(data_batch)
                if self.data.name == 'ali_home':
                    target = data_batch[-1]  # ali
                else:
                    target = data_batch[-2]  # amazon
                loss, score, pred = self.do_train(*data_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_avg += loss
                auc_avg += auc(score, target)
                f1_avg += f_score(pred, target)

            self.i_epoch += 1
            time_epoch = time.time() - time_start
            loss_avg = loss_avg / (i_batch + 1)
            auc_avg = auc_avg / (i_batch + 1)
            f1_avg = f1_avg / (i_batch + 1)

            print('{} {:<4d} | loss: {:+.4f}'.format(self.name, self.i_epoch, loss_avg), end='')
            try:
                print('| ui: {:.4f}, cate: {:.4f}'.format(self.attn_ui, self.attn_cate), end='')
            except:
                print(end='')
            print('| auc: {:+.4f}, f1: {:+.4f} | {:.2f}s'.format(auc_avg, f1_avg, time_epoch))
            if self.i_epoch % step == 0:
                self.flag_train = False

    def predict(self, data_test, key, ind_sel):
        with torch.no_grad():
            return getattr(self, 'predict_{}'.format(key))(data_test, ind_sel)

    def predict_topk_mix(self, data_test, ind_sel):
        """
        :return collect_nid_sort: T[n, len_mix]
        """
        batch_size = int(self.batch_size / 64)
        data_zip = [(uq, mix) for uq, mix in zip(*data_test)]
        data_iter = DataLoader(data_zip, batch_size=batch_size)
        nid_sort = []
        for i_batch, data_batch in enumerate(data_iter):
            data_batch = self.batch_to(data_batch)
            mix_batch = data_batch[1]
            k = mix_batch.shape[1]
            score_batch = self(*data_batch)
            rank_batch = score_batch.topk(k)[1]
            nid_sort_batch = smart_sort(mix_batch, rank_batch)
            nid_sort.append(nid_sort_batch)
        collect_nid_sort = torch.cat(nid_sort, dim=0)
        return collect_nid_sort

    @staticmethod
    def score_ctr_pred(score):
        return (score > 0.5).int()

    @timing
    def move_to(self, device):
        print('move {} to {}'.format(self.name, device))
        self.device = device
        self.tokens_q = self.tokens_q.to(device)
        self.lens_q = self.lens_q.to(device)
        self.tokens_i = self.tokens_i.to(device)
        self.lens_i = self.lens_i.to(device)
        self.to(device)

    def batch_to(self, data_batch):
        return tuple(arg.to(self.device) for arg in data_batch)
