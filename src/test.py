# coding: utf-8
from gensim.models import word2vec, fasttext, doc2vec, TfidfModel
from gensim.corpora import Dictionary
import os
import sys
import gensim
import random

def test(model_type, model_name):
    """
    モデルを読み込み各種処理を行う.
    """

    model = load_model(model_type, model_name)

    # 文章の処理
    #if model_type=="doc2vec":
    #    print("")
    # 単語の処理

    # show_wv(model)
    # show_dv(model_type, model)

def load_model(model_type, model_name):
    """
    モデルの読み込み
    """

    print("load model.")
    if(model_type=="word2vec"):
        model = word2vec.Word2Vec.load("model/%s" % model_name)
    elif(model_type=="doc2vec"):
        model = doc2vec.Doc2Vec.load("model/%s" % model_name)
    elif(model_type=="fasttext"):
        model = fasttext.FastText.load("model/%s" % model_name)
    elif(model_type=="bow"):
        # 要corpusの読み込み
        # corpus = [text.split() for text in texts]
        model = Dictionary.load("model/%s" % model_name)
    elif(model_type=="tfidf"):
        model = TfidfModel.load("model/%s" % model_name)

    return model

def show_wv(model, word=None):
    """
    単語ベクトルの表示
    
    word:
        None: 学習済みの重みからランダムに単語ベクトルを表示
        not None: 学習済みの重みから指定した単語ベクトルを表示
        ※ wordパラメータに単語を入力
    """

    print("show wordsvector.")
    if not word:
        word_keys = list(model.wv.vocab.keys())
        n = round(random.random() * len(word_keys))
        print(word_keys[n])
        print(model.wv.word_vec(word_keys[n]))
    else:
        print(word)
        print(model.wv.word_vec(word))        

def show_dv(model_type, model):
    """
    文章ベクトルの表示(bow, tfidf, doc2vec)
    文章ベクトルの計算法(要検討➞word2vec, fasttext)
    """

    print("show docsvector.")
    if model_type == "doc2vec" or model_type == "bow" or model_type == "tfidf":
        print("test")
    else:
        print("only doc2vec or bow or tfidf.")

def calc_sim_wv(wv1, wv2):
    """
    単語ベクトル間の類似度計算と表示

    wv1: 単語ベクトル
    wv2: 単語ベクトル
    """

def calc_sim_dv(dv1, dv2):
    """
    文章ベクトル間の類似度計算と表示

    dv1: 文書ベクトル
    dv2: 文書ベクトル
    """

def wordclowd():
    """
    文書・単語ベクトル間の類似度を可視化
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