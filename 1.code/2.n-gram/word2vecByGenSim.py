'''
Created on Jan 28, 2016

@author: jaeh
'''

from gensim.models import Word2Vec;
from gensim.models import KeyedVectors;
import torch
import torchwordemb

import logging, os;
import re
import time
init_time = time.time()

def measure():
    global init_time
    after_time = time.time()
    dif_time = after_time - init_time
    hour = int(dif_time / 3600)
    min = int((dif_time - hour * 3600) / 60)
    sec = dif_time - hour * 3600 - min * 60
    print('Processing Time: ' + str(hour) + "hour " + str(min) + "min " + str(sec) + "sec ")


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    re.sub: replace fuction.. reference: http://egloos.zum.com/sweeper/v/3065126
    """
    string = re.sub(r"[^A-Za-z0-9(),/+\-!?\'\`]", " ", string)  # punctuation -> whitespace
    string = re.sub(r"\'s", " \'s", string)                 # 's -> whitespace + 's
    string = re.sub(r"\'ve", " \'ve", string)               # 've -> whitespace + 've
    string = re.sub(r"n\'t", " n\'t", string)               # n't -> whitespace + n't
    string = re.sub(r"\'re", " \'re", string)               # 're -> whitespace + 're
    string = re.sub(r"\'d", " \'d", string)                 # 'd -> whitespace + 'd
    string = re.sub(r"\'ll", " \'ll", string)               # 'll -> whitespace + 'll
    string = re.sub(r",", " , ", string)                    # , -> whitespace + , + whitespace
    string = re.sub(r"!", " ! ", string)                    # ! -> whitespace + ! + whitespace
    string = re.sub(r"\(", " ( ", string)                   # ( -> whitespace + ( + whitespace
    string = re.sub(r"\)", " ) ", string)                   # ) -> whitespace + ) + whitespace
    string = re.sub(r"\?", " ? ", string)                   # ? -> whitespace + ? + whitespace
    string = re.sub(r"\s{2,}", " ", string)                 # consecutive whitespace -> single whitespace
    return string.strip() if TREC else string.strip().lower()


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname;
    
    def __iter__(self):
        for line in open(self.dirname, 'r'):
            line = line.replace("\n","");
            line = line.replace("\r","");
            yield clean_str(line).split();


# def from_pretrained(embeddings, freeze=True):
#     assert embeddings.dim() == 2, \
#          'Embeddings parameter is expected to be 2-dimensional'
#     # rows, cols = embeddings.shape
#     # embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
#     embedding = torch.nn.Embedding(num_embeddings=len(embeddings), embedding_dim=len(embeddings[0]))
#     embedding.weight = torch.nn.Parameter(embeddings)
#     embedding.weight.requires_grad = not freeze
#     return embedding


if __name__=="__main__":
    ## all PubMed based WE result ##

    logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = MySentences('../resource/GE,PC.txt')
    model = Word2Vec(sentences, size=200, alpha=0.05, sample=0.00001, negative=10, workers=30, sg=1, window=16, iter=50)
    model.save('../result/test.model')
    model.wv.save_word2vec_format('../result/test.tsv')
    model.wv.save_word2vec_format('../result/test.model.bin', '../result/test.voca', binary=True)

    # sentences = MySentences('/data2/djjang/CODA_TM/WordEmbedding/Resources/word2vec/Portion_2');
    # loadedModel = Word2Vec.load('../result/test.result')
    # loadedModel = KeyedVectors.load_word2vec_format('../result/test.result.bin', '../result/test.voca', binary=True)
    # weights = torch.FloatTensor(loadedModel.syn0)
    # embedding = from_pretrained(weights)
    # print(embedding)

    # moreSentences = MySentences('/data2/djjang/CODA_TM/WordEmbedding/Resources/word2vec/W2V_PubMedInput/Portion_2')
    # loadedModel.train(moreSentences)
    # loadedModel = word2vec.Word2Vec(moreSentences, size=200, workers=30)
    # loadedModel.save_word2vec_format('/data2/djjang/CODA_TM/WordEmbedding/Results/word2vec/Gensim_word2vec/PubMed50_D250_p12.result.bin', '/data2/djjang/CODA_TM/WordEmbedding/Results/word2vec/Gensim_word2vec/PubMed50_D250_p12.voca', binary=True)
    # loadedModel.save('/data2/djjang/CODA_TM/WordEmbedding/Results/word2vec/Gensim_word2vec/PubMed50_D200_p12.result')

    # result = loadedModel.similar_by_word("the")
    # print("{}: {:.4f}".format(*result[0]))

    # # sentences = word2vec.Text8Corpus('/data2/djjang/CODA_TM/WordEmbedding/Resources/word2vec/Gensim_word2vec/text8')
    # result['expression']
    measure()