# coding: utf-8
from gensim.models import word2vec, fasttext, doc2vec
import os
import sys
import gensim
import random

def test(model_type, model_name):
    """
    write codes.
    """

    model = load_model(model_type, model_name)
    show_wv(model)

def load_model(model_type, model_name):
    """
    モデルの読み込み
    """

    if(model_type=="word2vec"):
        model = word2vec.Word2Vec(model_name)
    elif(model_type=="doc2vec"):
        model = doc2vec.Doc2Vec(model_name)
    elif(model_type=="fasttext"):
        model = fasttext.FastText(model_name)

    return model

def show_wv(model):
    """
    単語ベクトルの表示
    """
    word_keys = list(model.wv.vocab.keys())
    print(model.wv.word_vec(word_keys[round(random.random() * len(word_keys))]))

def show_dv(model_type, model):
    """
    文章ベクトルの表示
    文章ベクトルの計算法(要検討)
    """

    if model_type == "doc2vec":
        
    else:
        print("only doc2vec.")

def calc_sim_wv():
    """
    単語ベクトル間の類似度計算と表示
    """

def calc_sim_dv():
    """
    文章ベクトル間の類似度計算と表示
    """

def wordclowd():
    """
    文書間類似度を可視化
    """

def main():
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数

    # デバッグプリント
    if(argc != 3):   # 引数が足りない場合は、その旨を表示
        print('Usage: # python %s model_type model_no' % argvs[0])
        quit()         # プログラムの終了

    model_type = argvs[1]
    model_no = argvs[2]
    model_name = "%s_%s.model" % (model_type, model_no)

    test(model_type, model_name)

if __name__ == "__main__":
    main()