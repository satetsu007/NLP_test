import MeCab
import os
import shutil
import sys
import re
import unicodedata
import nltk

def wakati(txt):
    """
    入力文章を分かち書きして出力
    """
    t = MeCab.Tagger("-Owakati")
    return t.parse(txt)

def wakati_remove_stopword(txt):
    """
    ストップワードを除去する

    記号や数字、助詞等
    """

def normalize(text):
    """
    文章の正規化
    """
    normalized_text = normalize_unicode(text)
    normalized_text = normalize_number(normalized_text)
    normalized_text = lower_text(normalized_text)
    return normalized_text

def lower_text(text):
    """
    A → a
    """
    return text.lower()

def normalize_unicode(text, form='NFKC'):
    """
    ｱｲｳｴｵ → アイウエオ
    """
    normalized_text = unicodedata.normalize(form, text)
    return normalized_text

def normalize_number(text):
    # 連続した数字を0で置換
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text