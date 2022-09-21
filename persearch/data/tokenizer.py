from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence

from persearch.utils import flatten


class Tokenizer(object):
    """ tokenize the given corpus
    def tokenize_<>(corpus, *args, **kwargs)
        - input
            - corpus: a list of textual documents (can split by space at first)
        - return
            - tokens: tensor [n_doc, max_len], padded tokens
            - lens: tensor [n_doc], lens of each doc
    """
    def __init__(self, mode: str = 'dict'):
        self.mode = mode  # (dict)
        self.tokenize = getattr(self, 'tokenize_{}'.format(self.mode))
        self.dictionary = None
        self.idx_pad = None

    def build_dictionary(self, corpus: list, sep: str = ' ') -> dict:
        """charge build-in dictionary"""
        corpus = [str(text).split(sep) for text in corpus]  # todo, modified pre-processing
        dictionary = list(set(flatten(corpus)))
        if '' in dictionary:
            dictionary.remove('')
        dictionary = [''] + dictionary
        self.dictionary = {w: i for i, w in enumerate(dictionary)}
        self.idx_pad = 0
        return self.dictionary

    def tokenize_dict(
            self, docs: list, dictionary: dict = None, pad_token: str = '',
            if_pad: bool = True, *args, **kwargs) -> ([torch.Tensor, list], torch.Tensor):
        """
        tokenize queries with dictionary
        :param docs: [n_doc], a list of documents
        :param dictionary: dict {token: index}
        :param pad_token: str
        :param if_pad: bool
        """
        lens = []
        docs_tokenized = []
        # initialize dictionary and padding idx
        if dictionary is not None:
            self.dictionary = dictionary
            self.idx_pad = self.dictionary[pad_token]

        pad_docs = [''] + docs
        # tokenize
        for query in tqdm(pad_docs):
            query_split = str(query).split(' ')
            query_tokenize = [self.dictionary[w] for w in query_split]
            docs_tokenized.append(torch.tensor(query_tokenize))
            lens.append(len(query_split))
        if if_pad:
            docs_pad = pad_sequence(
                docs_tokenized, batch_first=True, padding_value=self.idx_pad)
        else:
            docs_pad = docs_tokenized
        lens = torch.tensor(lens)
        return docs_pad, lens
