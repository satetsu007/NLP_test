# coding: utf-8
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
# from gensim.models.doc2vec import Doc2vec, TaggedDocument
import logging
import os
import numpy as np
import pandas as pd
import sys

def main():
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数
    # デバッグプリント
    if(argc != 2):   # 引数が足りない場合は、その旨を表示
        print('Usage: # python %s filename' % argvs[0])
        quit()         # プログラムの終了
    if(argvs[1]=="word2vec"):
        w2v()
    elif(argvs[1]=="doc2vec"):
        d2v()

def w2v():
    corpus_file = "data.txt"
    model_name = "data0.model"
    iter_count = 1000

    os.chdir("data")

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print("prepare data.")
    sentences = word2vec.LineSentence(corpus_file)
    print("train model.")
    # workers=1にしなければseed固定は意味がない(ドキュメントより)
    model = word2vec.Word2Vec(sentences, size=200, min_count=1, window=15, seed=1, workers=8, iter=iter_count)
    print("save model.")
    model.save("../model/%s" % model_name)

def d2v():
    """
    write code.
    """


if __name__=="__main__":
    main()