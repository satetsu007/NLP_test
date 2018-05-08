import MeCab
import os
import shutil
import sys
import re
import unicodedata
import nltk
from bs4 import BeautifulSoup

def main():
    """
    テキストの前処理を行う

    標準でmain, targetフォルダを読み込む
    """

    print("preset.")
    os.chdir("data")
    preset()

def preset():
    """
    テキストの前処理
    """

    m = "main"
    t = "target"

    main_folders = [m_f for m_f in os.listdir(m) if not m_f==".DS_Store"]
    target_folders = [t_f for t_f in os.listdir(t) if not t_f==".DS_Store"]
    
    main_files = [os.listdir("%s/%s" % (m, m_f)) for m_f in main_folders if os.path.isdir(("%s/%s") % (m, m_f))]
    target_files = [os.listdir("%s/%s" % (t, t_f)) for t_f in target_folders if os.path.isdir(("%s/%s") % (t, t_f))]

    for i, main_folder in enumerate(main_folders):
        for main_file in main_files[i]:
            if main_file[-4:] == ".txt" and not "wakati" in main_file:
                f = open("%s/%s/%s" % (m, main_folder, main_file), "r", encoding="utf-8")
                text = f.read()
                f.close()
                text = clean(text)
                text = normalize(text)
                text = wakati(text)
                f = open("%s/%s/%s" % (m, main_folder, main_file[:-4]+"_wakati"+main_file[-4:]), "w", encoding="utf-8")
                text = f.write(text)
                f.close()

    for i, target_folder in enumerate(target_folders):
        for target_file in target_files[i]:
            if target_file[-4:] == ".txt" and not "wakati" in target_file:
                f = open("%s/%s/%s" % (t, target_folder, target_file), "r", encoding="utf-8")
                text = f.read()
                f.close()
                text = clean(text)
                text = normalize(text)
                text = wakati(text)
                f = open("%s/%s/%s" % (t, target_folder, target_file[:-4]+"_wakati"+target_file[-4:]), "w", encoding="utf-8")
                text = f.write(text)
                f.close()


def clean(text):
    """
    """
    text = clean_text(text)
    return text

def clean_text(text):
    replaced_text = '\n'.join(s.strip() for s in text.splitlines()[2:] if s != '')  # skip header by [2:]
    replaced_text = re.sub(r'[【】]', ' ', replaced_text)       # 【】の除去
    replaced_text = re.sub(r'[（）()]', ' ', replaced_text)     # （）の除去
    replaced_text = re.sub(r'[［］\[\]]', ' ', replaced_text)   # ［］の除去
    replaced_text = re.sub(r'[@＠]\w+', '', replaced_text)  # メンションの除去(twitter等)
    replaced_text = re.sub(r'https?:\/\/.*?[\r\n ]', '', replaced_text)  # URLの除去
    replaced_text = re.sub(r'　', ' ', replaced_text)  # 全角空白の除去
    return replaced_text

def clean_html_tags(html_text):
    """
    htmlタグの除去
    """
    soup = BeautifulSoup(html_text, 'html.parser')
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text

def clean_html_and_js_tags(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    [x.extract() for x in soup.findAll(['script', 'style'])]
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text

def clean_url(html_text):
    """
    urlの除去
    """
    clean_text = re.sub(r'http\S+', '', html_text)
    return clean_text

def clean_code(html_text):
    """
    Qiitaのコードを取り除きます
    :param html_text:
    :return:
    """
    soup = BeautifulSoup(html_text, 'html.parser')
    [x.extract() for x in soup.findAll(class_="code-frame")]
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text

def wakati(text, mode="wakati", original=True):
    """
    入力文章を分かち書きして出力

    mode: wakati or morph
        wakati: 単純に分かち書き処理
        morph: 形態素の品詞を判別して不要な単語を除去 → 分かち書き処理
    original: True or False
        True: mode==morph時に単語の原形を取り出す
        False: mode==morph時に文中の単語を取り出す
    """
    if mode=="wakati":
        t = MeCab.Tagger("-Owakati")
        return t.parse(text)
    elif mode=="morph":
        t = MeCab.Tagger("")
        # 抜き出す品詞リスト
        parts_of_speech = ["名詞", "動詞", "形容詞", "形容動詞"]
        if original:
            morphs = [[line.split(",")[-3], line.split(",")[0].split("\t")[-1]] for line in t.parse(text).split("\n")[:-2]]
        else:
            morphs = [line.split(",")[0].split("\t") for line in t.parse(text).split("\n")[:-2]]
        morphs = [morph[0] for morph in morphs if morph[1] in parts_of_speech]

        tmp = ""
        for morph in morphs:
            tmp += morph + " "
        return tmp

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
    """
    連続する数字を0に置換する
    """
    # 連続した数字を0で置換
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text

if __name__ == "__main__":
    main()