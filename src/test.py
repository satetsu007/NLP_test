# coding: utf-8
from gensim.models import word2vec, fasttext, doc2vec, TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
import os
import sys
import gensim
import random
import numpy as np
import pandas as pd
from module import bow_read_docs, vec2dense

def test(model_type, model_name):
    """
    モデルを読み込み各種処理を行う.
    """

    # modelを読み込む
    # 単語, 文章ベクトルをcsv形式で保存する
    if(model_type=="bow"):
        dic = load_model(model_type, model_name)
        corpus = bow_read_docs("data/tmp_file")
        save_vector(model_type, model_name, model=None, dic=dic, corpus=corpus)
    elif(model_type=="tfidf"):
        model = load_model(model_type, model_name)
        dic = Dictionary.load("model/bow_%s" % model_name[-7:])
        corpus = bow_read_docs("data/tmp_file")
        save_vector(model_type, model_name, model=model, dic=dic, corpus=corpus)
    elif(model_type in ["doc2vec", "word2vec", "fasttext"]):
        model = load_model(model_type, model_name)
        save_vector(model_type, model_name, model=model)
    elif(model_type=="all"):
        tmp_name = model_name[-7:]
        model_name = "word2vec_%s" % tmp_name
        model_type = model_name[:-8]
        model = load_model(model_type, model_name)
        save_vector(model_type, model_name, model=model)
        model_name = "fasttext_%s" % tmp_name
        model_type = model_name[:-8]
        model = load_model(model_type, model_name)
        save_vector(model_type, model_name, model=model)
        model_name = "doc2vec_%s" % tmp_name
        model_type = model_name[:-8]
        model = load_model(model_type, model_name)
        save_vector(model_type, model_name, model=model)
        model_name = "bow_%s" % tmp_name
        model_type = model_name[:-8]
        dic = load_model(model_type, model_name)
        corpus = bow_read_docs("data/tmp_file")
        save_vector(model_type, model_name, model=None, dic=dic, corpus=corpus)
        model_name = "tfidf_%s" % tmp_name
        model_type = model_name[:-8]
        model = load_model(model_type, model_name)
        dic = Dictionary.load("model/bow_%s" % model_name[-7:])
        corpus = bow_read_docs("data/tmp_file")
        save_vector(model_type, model_name, model=model, dic=dic, corpus=corpus)

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

def save_vector(model_type, model_name, model=None, dic=None, corpus=None):
    """
    単語, 文章ベクトルを保存

    dic, corpusはbow or tfidf時に必要
    """

    print("save %s vector." % model_type)
    file_names = os.listdir("data/tmp_file")
    
    if(model_type=="word2vec"):
        df = pd.DataFrame(model.wv.vectors)
        df.index = model.wv.vocab.keys()
        save_df(df, model_name, mode="word")
    elif(model_type=="doc2vec"):
        df = pd.DataFrame(model.wv.vectors)
        df.index = model.wv.vocab.keys()
        save_df(df, model_name, mode="word")
        df2 = pd.DataFrame(np.array([model.docvecs[i] for i, _ in enumerate(file_names)]))
        df2.index = file_names
        save_df(df2, model_name, mode="doc")
    elif(model_type=="fasttext"):
        df = pd.DataFrame(model.wv.vectors)
        df.index = model.wv.vocab.keys()
        save_df(df, model_name, mode="word")
    elif(model_type=="bow"):
        bow_matrix = np.array([vec2dense(dic.doc2bow(corpus[i]),len(dic)) for i in range(len(corpus))])
        df = pd.DataFrame(bow_matrix)
        df.index = file_names
        df.columns = dic.token2id.keys()
        save_df(df, model_name, mode="doc")
    elif(model_type=="tfidf"):
        bow_corpus = [dic.doc2bow(d) for d in corpus]
        model = model[bow_corpus]
        df = pd.DataFrame(np.array([vec2dense(doc,len(dic)) for doc in model]))
        df.index = file_names
        df.columns = dic.token2id.keys()
        save_df(df, model_name, mode="doc")

def save_df(df, model_name, mode="word"):
    """
    dfの保存
    
    df: pandasのDataFrame形式
    mode: word or doc
        保存先(dfが文章ベクトルか単語ベクトルか)を決める
    """

    df.to_csv("data/csv/vector/%s/%s.csv" % (mode, model_name[:-6]))

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