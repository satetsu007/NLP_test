from gensim.models import doc2vec
import os
import sys
import logging
import gensim
import numpy as np
import pandas as pd
import shutil
from module import swap, read_docs

def train(model_type, model_name):
    """
    gensimを使用して単語ベクトルを学習, モデルの保存を行う.
    各種学習アルゴリズムは下記関数にて呼び出す.
    
    doc2vec: d2v

    実行例(fasttextを使用)
    python nlp_with_gensim.py fasttext
    """

    # corpus_file = "data.txt"
    iter_count = 1

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if(model_type=="doc2vec"):
        d2v(model_name, iter_count)
    else:
        print("only doc2vec.")

def set_data():
    main_folders = os.listdir("main")
    target_folders = os.listdir("target")

    main_files = [os.listdir("main/%s" % m_f) for m_f in main_folders]
    target_files = [os.listdir("target/%s" % t_f) for t_f in target_folders]

    for i, main_folder in enumerate(main_folders):
        for main_file in main_files[i]:
            shutil.copy("main/%s/%s" % (main_folder, main_file), "tmp_file/m:%s_%s"  % (main_folder, main_file))

    for i, target_folder in enumerate(target_folders):
        for target_file in target_files[i]:
            shutil.copy("target/%s/%s" % (target_folder, target_file), "tmp_file/t:%s_%s"  % (target_folder, target_file))

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

def test(model_type, model_name):
    """
    各種ファイルの類似度を計算
    計算結果を出力

    保存ファイル例
    data/csv/doc2vec_tag_1.csv
    """

    print("load model.")
    model = load_model(model_type, model_name)
    print("calc simirality.")
    df, model = calc(model)
    print("save csv.")
    save_csv(df, model_name)

def load_model(model_type, model_name):
    """
    モデルの読み込み
    """

    print("load model.")
    if(model_type=="doc2vec"):
        model = doc2vec.Doc2Vec.load("model/%s" % model_name)

    return model

def calc_sim_dv(model, dv1, dv2):
    """
    文章ベクトル間の類似度計算と表示

    dv1: 文書タグ
    dv2: 文書タグ
    """
    return model.docvecs.similarity(dv1, dv2)

def calc(model):
    model.docvecs.doctags = os.listdir("data/tmp_file")
    sim = []
    for i in range(model.docvecs.count):
        tmp = []
        for j in range(model.docvecs.count):
            tmp.append(calc_sim_dv(model, i, j))
        sim.append(tmp)
    
    sim = np.array(sim)
    df = pd.DataFrame(sim)
    df.index = model.docvecs.doctags
    df.columns = model.docvecs.doctags
    
    return df, model

def save_csv(df, model_name):
    """
    計算結果の出力
    """
    df.to_csv("data/csv/%s.csv" % model_name[:-6])

def main():
    """
    gensimを使用して単語ベクトルを学習, モデルの保存を行う.
    各種学習アルゴリズムは下記関数にて呼び出す.
    
    doc2vec: d2v

    NLP_testフォルダから実行すること

    実行例(doc2vecを使用して学習)
    python nlp_with_gensim.py train doc2vec 1
    """

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数
    
    # デバッグプリント
    if(argc != 4):   # 引数が足りない場合は、その旨を表示
        print('Usage: # python %s train/test model_type model_no' % argvs[0])
        quit()         # プログラムの終了

    model_type = argvs[2]
    model_no = argvs[3]
    model_name = "%s_tag_%s.model" % (model_type, model_no)

    if(argvs[1]=="train"):
        train(model_type, model_name)
    elif(argvs[1]=="test"):
        test(model_type, model_name)

if __name__=="__main__":
    main()