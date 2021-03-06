from gensim.models import word2vec
import os
import logging

def main():
    corpus_file = "data.txt"
    iter_count = 1

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    f = open("data/%s" % corpus_file,  "r")
    text = f.read()
    sentences = [s.split(" ") for s in text.split("\n")]

    for i in range(1, 2):
        model = word2vec.Word2Vec(min_count=1, seed=1, workers=1, iter=iter_count)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        model.save("model/word2vec_gensim_iter=1_%s.model" % i)

if __name__ == "__main__":
    main()