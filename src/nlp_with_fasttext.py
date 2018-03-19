from gensim.models import fasttext
import os
import logging

os.getcwd()
os.chdir('/media/satetsu-gpu/Data/program/project/NLP/NLP_test')

corpus_file = "data.txt"
iter_count = 100

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

f = open("data/%s" % corpus_file,  "r")
text = f.read()
sentences = [s.split(" ") for s in text.split("\n")]

for i in range(5):
    model = fasttext.FastText(min_count=1, seed=1, workers=1, iter=iter_count)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.save("model/fasttext_gensim_iter=100_%s.model" % i)

