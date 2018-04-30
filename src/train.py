# coding: utf-8
# coding: utf-8
from gensim.models import word2vec, fasttext, doc2vec, TfidfModel
import logging
import os
import sys
import gensim
import smart_open
from module import set_data, read_docs
from gensim.corpora import Dictionary

def train(model_type, model_name):
    """
    gensimを使用して単語ベクトルを学習, モデルの保存を行う.
    各種学習アルゴリズムは下記関数にて呼び出す.
    
    単語→ベクトル化
    標準でdata.txtを読み込む

    word2vec: w2v
    fasttext: ft
    
    文章→ベクトル化
    標準でmain, targetフォルダを読み込む
    doc2vec: d2v
    bow(bag-of-words): bow
    tfidf(tf-idf): tfidf

    NLP_testフォルダから実行すること

    実行例(fasttextを使用して学習)
    python nlp_with_gensim.py fasttext 1
    """

    corpus_file = "data.txt"
    iter_count = 1

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if(model_type=="word2vec"):
        w2v(corpus_file, model_name, iter_count)
    elif(model_type=="fasttext"):
        ft(corpus_file, model_name, iter_count)
    elif(model_type=="doc2vec"):
        d2v(model_name, iter_count)
    elif(model_type=="bow"):
        bow(model_name)
    elif(model_type=="tfidf"):
        tfidf(model_name)

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
    f = open("%s" % corpus_file,  "r", encoding="utf-8")
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

def d2v(model_name, iter_count):
    """
    doc2vec
    """

    print("prepare data.")
    os.chdir("data")
    set_data()
    sentences = list(read_docs())

    print("train model.")
    # workers=1にしなければseed固定は意味がない(ドキュメントより)
    model = doc2vec.Doc2Vec(min_count=1, seed=1, workers=1, iter=iter_count)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

    print("save model.")
    os.chdir("..")
    model.save("model/%s" % model_name)

def bow(model_name):
    """
    bag-of-words
    """

    print("prepare data.")
    os.chdir("data")
    set_data()
    sentences = read_docs(mode="bow")

def tfidf(model_name):
    """
    tf-idf
    """
    print("prepare data.")
    os.chdir("data")
    set_data()
    sentences = read_docs(mode="tfidf")

    dic = Dictionary(sentences)
    ## 「出現頻度が20未満の単語」と「30%以上の文書で出現する単語」を排除
    ## dic.filter_extremes(no_below = 20, no_above = 0.3)
    bow_corpus = [dic.doc2bow(d) for d in sentences]

    model = TfidfModel(bow_corpus)

    print("save model.")
    os.chdir("..")
    model.save("model/%s" % model_name)

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

    train(model_type, model_name)

if __name__ == "__main__":
    main()