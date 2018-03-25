# coding: utf-8
# coding: utf-8
from gensim.models import word2vec, fasttext
# from gensim.models.doc2vec import Doc2vec, TaggedDocument
import logging
import os
import numpy as np
import pandas as pd

def train(model_type):
    """
    gensimを使用して単語ベクトルを学習, モデルの保存を行う.
    各種学習アルゴリズムは下記関数にて呼び出す.
    
    word2vec: w2v
    doc2vec: d2v
    fasttext: ft

    実行例(fasttextを使用)
    python nlp_with_gensim.py fasttext
    """
    os.chdir('/media/satetsu-gpu/Data/program/project/NLP/NLP_test')

    corpus_file = "data.txt"
    model_name = "data0.model"
    iter_count = 3

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if(model_type=="word2vec"):
        w2v(corpus_file, model_name, iter_count)
    elif(model_type=="doc2vec"):
        d2v(corpus_file, model_name, iter_count)
    elif(model_type=="fasttext"):
        ft(corpus_file, model_name, iter_count)

def w2v(corpus_file, model_name, iter_count):
    """
    word2vec
    """
    print("prepare data.")
    os.chdir("data")
    sentences = word2vec.LineSentence(corpus_file)

    print("train model.")
    # workers=1にしなければseed固定は意味がない(ドキュメントより)
    model = word2vec.Word2Vec(sentences, min_count=1, seed=1, workers=1, iter=iter_count)
    
    print("save model.")
    os.chdir("..")
    model.save("model/%s" % model_name)

def ft(corpus_file, model_name, iter_count):
    """
    fasttext
    """
    print("prepare data.")
    os.chdir("data")
    f = open("%s" % corpus_file,  "r")
    text = f.read()
    sentences = [s.split(" ") for s in text.split("\n")]
    
    print("train model.")
    # workers=1にしなければseed固定は意味がない(ドキュメントより)
    model = fasttext.FastText(min_count=1, seed=1, workers=1, iter=iter_count)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

    print("save model.")
    os.chdir("..")
    model.save("model/%s" % model_name)

def d2v(corpus_file, model_name, iter_count):
    """
    write code.
    """


if __name__=="__main__":
    model_type = "fasttext"
    train(model_type)