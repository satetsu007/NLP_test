import os
from gensim.models import word2vec, fasttext

#model = word2vec.Word2Vec.load("../model/data.model")

os.chdir("src")

model_type = "word2vec"

if model_type == "word2vec":
    model1 = word2vec.Word2Vec.load("../model/word2vec_gensim_iter=1_0.model")
    model2 = word2vec.Word2Vec.load("../model/word2vec_gensim_iter=1_0.model")
    print(model1.wv.word_vec("伊藤忠商事"))
    print(model2.wv.word_vec("伊藤忠商事"))

elif model_type == "fasttext":
    model1 = fasttext.FastText.load("../model/fasttext_gensim_iter=100_0.model")
    model2 = fasttext.FastText.load("../model/fasttext_gensim_iter=100_1.model")

    print(model1.wv.word_vec("伊藤忠商事"))
    print(model2.wv.word_vec("伊藤忠商事"))
    print(model3.wv.word_vec("伊藤忠商事"))
    print(model4.wv.word_vec("伊藤忠商事"))
    print(model5.wv.word_vec("伊藤忠商事"))