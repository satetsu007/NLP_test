from gensim.models import doc2vec
import os
import logging
import gensim
from module import read_corpus


def main():
    corpus_file = "data.txt"
    iter_count = 1

    sentences = list(read_corpus(corpus_file))

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    for i in range(1):
        model = doc2vec.Doc2Vec(min_count=1, seed=1, workers=1, iter=iter_count)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        model.save("model/doc2vec_gensim_iter=1_%s.model" % i)

if __name__ == "__main__":
    main()
