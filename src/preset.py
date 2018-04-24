import MeCab
import os
import shutil
import sys

def wakati(txt):
    """
    入力文章を分かち書きして出力
    """
    t = MeCab.Tagger("-Owakati")
    return t.parse(txt)

def wakati_without_stopword(txt):
    """
    """