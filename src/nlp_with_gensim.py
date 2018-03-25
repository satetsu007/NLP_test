# coding: utf-8
from train import train
from test import test
from gensim.models import word2vec, fasttext
# from gensim.models.doc2vec import Doc2vec, TaggedDocument
import logging
import os
import numpy as np
import pandas as pd
import sys

def main():
    """
    gensimを使用して単語ベクトルを学習, モデルの保存を行う.
    各種学習アルゴリズムは下記関数にて呼び出す.
    
    word2vec: w2v
    doc2vec: d2v
    fasttext: ft

    実行例(fasttextを使用)
    python nlp_with_gensim.py fasttext
    """
    os.chdir('/media/satetsu-gpu/Data/program/project/NLP/NLP_test')

    model_type = "fasttext"

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数
    # デバッグプリント
    if(argc != 2):   # 引数が足りない場合は、その旨を表示
        print('Usage: # python %s filename' % argvs[0])
        quit()         # プログラムの終了
    if(argvs[1]=="train"):
        train(model_type)
    elif(argvs[1]=="test"):
        test(model_type)

if __name__=="__main__":
    main()