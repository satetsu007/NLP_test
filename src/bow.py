import numpy as np
import os
import pandas as pd
from nlp_with_tag import set_data
from sklearn.feature_extraction.text import CountVectorizer as CV

def read_docs(folder_name="tmp_file"):
    fnames = os.listdir(folder_name)
    corpus_list = []
    for fname in fnames:
        f = open("%s/%s" % (folder_name, fname))
        txt = f.read()
        corpus_list.append(txt)
    return corpus_list

def text2bow(corpus_list):
    cv = CV()
    mat = cv.fit_transform(corpus_list).todense()
    vocab = cv.vocabulary_
    return mat, vocab

def bow2df(mat, vocab, folder_name="tmp_file"):
    df = pd.DataFrame(mat)
    df.columns = vocab.keys()
    file_list = os.listdir(folder_name)
    df.index = file_list
    return df

def bow(model_name):
    """
    Bag of Words
    """
    print("prepare data.")
    os.chdir("data")
    set_data()

    print("train model.")
    corpus_list = read_docs()
    matrix, vocab = text2bow(corpus_list)
    df = bow2df(matrix, vocab)

    print("save model.")
    df.to_csv("../model/%s" % model_name)

