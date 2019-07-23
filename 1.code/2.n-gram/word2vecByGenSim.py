from gensim.models import Word2Vec
import logging
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


def clean_str(string):
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
    return string.strip().lower()


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    
    def __iter__(self):
        for line in open(self.dirname, 'r'):
            line = line.replace("\n","")
            line = line.replace("\r","")
            yield clean_str(line).split()

if __name__=="__main__":
    logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = MySentences('corpus.txt')
    model = Word2Vec(sentences, size=200, alpha=0.05, sample=0.00001, negative=10, workers=30, sg=1, window=16, iter=50)
    model.wv.save_word2vec_format('n-gram.model.bin', 'output.voca', binary=True)
    measure()