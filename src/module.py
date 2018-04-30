# coding: utf-8

from gensim.models import word2vec, fasttext, doc2vec
from test import load_model, show_wv
import gensim
import smart_open
import os
import shutil

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

def read_docs(mode="doc2vec", folder_name="tmp_file", tokens_only=False):
    fnames = os.listdir(folder_name)
    if mode=="doc2vec":
        for i, fname in enumerate(fnames):
            f = open("%s/%s" % (folder_name, fname))
            txt = f.read()
            if tokens_only:
                yield gensim.utils.simple_preprocess(txt)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(txt), [i])
    elif mode=="bow" or mode=="tfidf":
        corpus_list = []
        for fname in fnames:
            f = open("%s/%s" % (folder_name, fname))
            txt = f.read()
            corpus_list.append(txt)
        return corpus_list

def set_data():
    m = "main"
    t = "target"
    main_folders = [m_f for m_f in os.listdir(m) if not m_f==".DS_Store"]
    target_folders = [t_f for t_f in os.listdir(t) if not t_f==".DS_Store"]

    main_files = [os.listdir("%s/%s" % (m, m_f)) for m_f in main_folders if os.path.isdir(("%s/%s") % (m, m_f))]
    target_files = [os.listdir("%s/%s" % (t, t_f)) for t_f in target_folders if os.path.isdir(("%s/%s") % (t, t_f))]

    for i, main_folder in enumerate(main_folders):
        for main_file in main_files[i]:
            if main_file[-4:] == ".txt":
                shutil.copy("%s/%s/%s" % (m, main_folder, main_file), "tmp_file/m:%s_%s"  % (main_folder, main_file))

    for i, target_folder in enumerate(target_folders):
        for target_file in target_files[i]:
            if main_file[-4:] == ".txt":
                shutil.copy("%s/%s/%s" % (t, target_folder, target_file), "tmp_file/t:%s_%s"  % (target_folder, target_file))

def to_fname(i, folder_name="tmp_file"):
    return os.listdir(folder_name)[i]
    
def to_iter(fname, folder_name="tmp_file"):
    return os.listdir(folder_name).index(fname)

def swap(var):
    if type(var) == int:
        return to_fname(var)
    elif type(var) == str:
        return to_iter(var)