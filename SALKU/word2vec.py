import os
import pandas as pd
import random
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from data_preprocess import getData
class word2vec:

    def train(self, word_dim):
        if not os.path.exists('data/processed_train_data.csv'):
            getData('train')
        data = pd.read_csv('data/processed_train_data.csv')
        sentences = []
        for t_sentence in data.loc[:,['sentence']].values:
            sentences.append(str(t_sentence[0]).split(' '))
        random.seed(10)
        sentences = random.sample(sentences, 100000)
        word2vec = Word2Vec(sentences, sg=1, size=word_dim, min_count=20)
        wv = word2vec.wv
        wv.save(os.path.join('word2vec/w2v' + str(word_dim) + '.model'))

    @classmethod
    def getWV(cls, word_dim):
        if not os.path.exists('word2vec/w2v' + str(word_dim) + '.model'):
            cls().train(word_dim)
        wv = KeyedVectors.load(os.path.join('word2vec/w2v' + str(word_dim) + '.model'))
        index2word = wv.index_to_key
        index2word.insert(0, '<!-PAD&UNK-!>')
        word2index = {word: index for index, word in enumerate(index2word)}
        index2vec = wv.vectors
        temp_array = np.zeros(shape=(1, word_dim), dtype=np.float)
        index2vec = np.insert(index2vec, 0, temp_array, axis=0)
        return index2word, word2index, index2vec