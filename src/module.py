# coding: utf-8

from gensim.models import word2vec, fasttext, doc2vec
from test import load_model, show_wv
import gensim
import smart_open

"""
実装済み関数
"""

def compare(model_no, word=None):
    """
    各種アルゴリズムでの学習結果を比較する.

    word: 比較したい学習済みの単語
    """

    print("compare vector.")
    model_types = ["word2vec", "doc2vec", "fasttext"]
    model_names = ["%s_%s.model" % (model_types[0], model_no),
                   "%s_%s.model" % (model_types[1], model_no),
                   "%s_%s.model" % (model_types[2], model_no)]

    for model_type, model_name in zip(model_types, model_names):
        print(model_type)
        model = load_model(model_type, model_name)
        show_wv(model, word=word)

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])