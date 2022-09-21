import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from persearch.utils import neg_sampling


class SkipGramLoss(nn.Module):
    """
    input
        pos: T[n, 1, d]
        neg: T[n, n_neg, d]
        context: T[n, 2*size_win, d]
    return: loss
    """
    def __init__(self):
        super(SkipGramLoss, self).__init__()
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, pos, neg, ctx):
        pos_expand = pos.expand_as(ctx)  # [n, 2size_win, d]
        dot_pos = pos_expand.mul(ctx).sum(-1)  # [n, 2size_win]
        score_pos = self.log_sigmoid(dot_pos)  # [n, 2size_win]

        dot_neg = torch.bmm(neg, ctx.transpose(1, 2))  # [n, n_neg, 2size_win]
        score_neg = self.log_sigmoid(-1 * dot_neg)

        n = score_pos.nelement() + score_neg.nelement()
        return -1 * (torch.sum(score_pos) + torch.sum(score_neg)) / n


class ParagraphVectorLoss(nn.Module):
    """
    input
        emb_entity: T[n, d]
        token_pos: T[n, pad_len]
    return loss
    """
    def __init__(self, emb_term, distribution, n_neg=5, subtract=True, max_len=256):
        super(ParagraphVectorLoss, self).__init__()
        self.emb_v = emb_term
        self.distribution = distribution
        self.n_v = self.distribution.shape[0]
        self.n_neg = n_neg
        self.subtract = subtract
        self.log_sigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()
        self.max_len = max_len

    def forward(self, emb_e, token_pos):
        bsz, pad_len = token_pos.shape
        if pad_len > self.max_len:
            pad_len = self.max_len
            token_pos = token_pos[:, :pad_len]
        lens = (token_pos != 0).sum(dim=1)
        n_token = (self.n_neg + 1) * lens.sum()
        token_neg = neg_sampling(
            token_pos, n_vocab=self.n_v, n_neg=pad_len * self.n_neg,
            weight=self.distribution, subtract=False, replace=True, # batch_size=64,
        )
        emb_v_pos = self.emb_v(token_pos)
        emb_v_neg = self.emb_v(token_neg)
        emb_v = torch.cat((emb_v_pos, -emb_v_neg), dim=1)  # [bsz, max_len(k+1), dim]
        emb_expand = emb_e.unsqueeze(1).expand(bsz, pad_len * (self.n_neg + 1), -1)
        dot = emb_expand.mul(emb_v).sum(-1)
        loss = - self.log_sigmoid(dot).sum() / n_token
        return loss


class ParagraphVectorDBOW(nn.Module):
    def __init__(self, emb_term, distribution, n_neg=5, subtract=True,
                 ):
        super(ParagraphVectorDBOW, self).__init__()
        self.loss_pv = ParagraphVectorLoss(
            emb_term, distribution=distribution,
            n_neg=n_neg, subtract=subtract)

    def forward(self, emb_e, tokens):
        return self.loss_pv(emb_e, tokens)

    def fit(self, emb_e, tokens, epoch=60, bsz=2048, lr=1e-3, weight_decay=1e-5):
        """"""
        print('> fit Paragraph VectorDBOW...')
        self.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = optim.Adam(
            [param for param in self.parameters()] + [emb_e],
            lr, weight_decay=weight_decay)
        dataset = TensorDataset(emb_e, tokens)
        loader = DataLoader(dataset, batch_size=bsz, shuffle=True)
        for i_epoch in range(epoch):
            loss_avg = 0.0
            time_start = time.time()
            for i_batch, (e, token) in enumerate(loader):
                e, token = e.to(device), token.to(device)
                loss = self(e, token)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_avg += loss
            loss_avg = loss_avg / (i_batch + 1)
            time_epoch = time.time() - time_start
            print('\r >>>> pv {:<4d} loss: {:.4f}, {:.2f}s'.format(
                i_epoch + 1, loss_avg, time_epoch), end='')
            print('> fit PV fin.')

    def predict(self, tokens):
        """"""

