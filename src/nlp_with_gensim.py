# coding: utf-8
from train import train
from test import test
from gensim.models import word2vec, fasttext
import logging
import os
import sys

def main():
    """
    gensimを使用して単語ベクトルを学習, モデルの保存を行う.
    各種学習アルゴリズムは下記関数にて呼び出す.
    
    word2vec: w2v
    doc2vec: d2v
    fasttext: ft

    NLP_testフォルダから実行すること

    実行例(fasttextを使用して学習)
    python nlp_with_gensim.py train fasttext 1
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
    model_name = "%s_%s.model" % (model_type, model_no)

    if(argvs[1]=="train"):
        train(model_type, model_name)
    elif(argvs[1]=="test"):
        test(model_type, model_name)

if __name__=="__main__":
    main()