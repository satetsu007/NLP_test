import os
from gensim.models import word2vec

#model = word2vec.Word2Vec.load("../model/data.model")
model1 = word2vec.Word2Vec.load("../model/data_light0.model")
model2 = word2vec.Word2Vec.load("../model/data_light1.model")
model3 = word2vec.Word2Vec.load("../model/data_light2.model")

print(model1.wv.word_vec("伊藤忠商事"))
print(model2.wv.word_vec("伊藤忠商事"))
print(model3.wv.word_vec("伊藤忠商事"))