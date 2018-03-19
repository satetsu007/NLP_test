# coding: utf-8
from gensim.models import word2vec, fasttext
# from gensim.models.doc2vec import Doc2vec, TaggedDocument
import logging
import os
import numpy as np
import pandas as pd
import sys

def main():
    """
    各種アルゴリズムでの学習結果を比較する.
    ある単語ベクトルの重み, ある単語ベクトルに類似するベクトル上位を表示する.
    """
    os.chdir('/media/satetsu-gpu/Data/program/project/NLP/NLP_test')
    
    print("load model.")
    os.chdir("model")

    print("compare vector.")
    