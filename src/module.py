# coding: utf-8

from gensim.models import word2vec, fasttext, doc2vec
from gensim.matutils import corpus2dense
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

def d2v_read_docs(folder_name, tokens_only):
    fnames = os.listdir(folder_name)
    for i, fname in enumerate(fnames):
        f = open("%s/%s" % (folder_name, fname))
        txt = f.read()
        if tokens_only:
            yield gensim.utils.simple_preprocess(txt)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(txt), [i])

def bow_read_docs(folder_name):
    fnames = os.listdir(folder_name)
    corpus_list = []
    for fname in fnames:
        f = open("%s/%s" % (folder_name, fname))
        txt = f.read()
        corpus_list.append(txt.split())
    return corpus_list

def read_docs(mode="doc2vec", folder_name="tmp_file", tokens_only=False):
    if mode=="doc2vec":
        d2v_read_docs(folder_name, tokens_only)
    elif mode=="bow":
        return bow_read_docs(folder_name)

def bow2vec(vec, num_terms):
    return list(corpus2dense([vec], num_terms=num_terms).T[0])

def set_data(mode="word"):
    """
    学習データの準備を行う

    mode: word or doc
        word: word2vec, fasttext時に選択
        doc: doc2vec, bow, tfidf時に選択
    """
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
    
    if mode=="word":
        tmp_file_list = os.listdir("tmp_file")
        tmp_txt = ""
        for tmp_file in tmp_file_list:
            f = open("tmp_file/%s" % tmp_file, "r", encoding="utf-8")
            txt = f.read()
            f.close()
            tmp_txt += txt + "\n"
        f = open("tmp.txt", "w", encoding="utf-8")
        f.write(tmp_txt)

def to_fname(i, folder_name="tmp_file"):
    return os.listdir(folder_name)[i]
    
def to_iter(fname, folder_name="tmp_file"):
    return os.listdir(folder_name).index(fname)

def swap(var):
    if type(var) == int:
        return to_fname(var)
    elif type(var) == str:
        return to_iter(var)