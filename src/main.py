# coding: utf-8
from gensim.models import word2vec
import logging
import os
import numpy as np

corpus = "data_light.txt"
model_name = "data_light0.model"

os.chdir("data")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("prepare data.")
sentences = word2vec.LineSentence(corpus)
print("train model.")
model = word2vec.Word2Vec(sentences, size=200, min_count=1, window=15, seed=1, workers=1)
print("save model.")
model.save("../model/%s" % model_name)
