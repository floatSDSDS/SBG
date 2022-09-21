from tqdm import tqdm
import numpy as np


class PrepareEmb(object):
    """prepare pre-trained embedding (subset) for a given corpus"""
    def __init__(self, name=''):
        self.name = name
        self.corpus = None
        self.vocab = None  # dict

    def restrict_w2v(self, w2v):
        """
        subset the input w2v with restriced_word_set
        :param w2v: gensim.models.keyedvectors
        """
        print('restrict w2v')
        new_vectors = []
        new_vocab = {}
        new_index2entity = []

        for i in tqdm(range(len(w2v.vocab))):
            word = w2v.index2entity[i]
            vec = w2v.vectors[i]
            vocab = w2v.vocab[word]
            if word in self.vocab.keys():
                vocab.index = len(new_index2entity)
                new_index2entity.append(word)
                new_vocab[word] = vocab
                new_vectors.append(vec)

        w2v.vocab = new_vocab
        w2v.vectors = np.array(new_vectors)
        w2v.index2entity = new_index2entity
        w2v.index2word = new_index2entity
        w2v.vectors_norm = None
