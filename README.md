# 文章, 単語のベクトル化プログラム

## 実装環境(実行環境)

python == 3.5 or 3.6<br>
tensorflow == 1.3.0<br>
gensim == 3.4.0<br>
fasttext == 0.8.3

***

## 実行順序

前処理 → 本処理 (→ 各種アルゴリズムの比較等)

1. preset.pyを実行
2. nlp_with_gensim.py train を実行
3. nlp_with_gensim.py test を実行

***

##  過去実装分との違い

- 実装難易度・可読性の向上
    - 実装時に使用するライブラリをgensimで統一
    - 実装環境(各種ライブラリのバージョン等)の統一
- MeCabを未インストールの端末でも実行可能に
    - 前処理と実行ファイルを切り分けた
- 様々なベクトル化アルゴリズムを取り入れた
    - doc2vec
    - bow
    - tfidf
    - word2vec
    - fasttext
- 処理速度の向上
    - 無駄な処理の削減, 効率的なプログラム(既存のライブラリ使用等)の実装
- 不要ファイルの排除
- コメント, 使用方法等の拡充

***

## 各種説明

### data

学習用データを保存

***

### ipynb

jupyter notebookを保存

***

### src

- bow.py
- compare.py
- module.py

- nlp_with_doc2vec.py
- nlp_with_fasttext.py
- nlp_with_gensim.py
- nlp_with_tag.py
- nlp_with_tensorflow.py
- nlp_with_word2vec.py

- preset.py
- test.py
- tfidf.py
- train.py
- weight.py


***

### model

学習済みの重みを保存

***

## ファイル説明

### src/train.py

#### train

#### w2v

#### ft

#### d2v

#### bow

#### tfidf

#### main

***

### src/test.py

#### test

#### load_model

#### show_wv

#### show_dv

#### calc_sim_wv

#### calc_sim_dv

#### wordclowd

***

### src/nlp_with_gensim.py

実行ファイル

***

### 読むべきもの

gensim: https://radimrehurek.com/gensim/<br>
tensorflow: https://www.tensorflow.org/<br>

***

## やったこと

- データを整理
- テキストファイルとデータフレームを結合
- テキストファイルを分かち書き
- テキストファイルを結合(light, middle, allの3パターン)
- 各種アルゴリズムを使用したtrain/testファイルの作成
- word2vec(with tensorflow)
- doc2vec(過去実装分)のdoctag考慮ver
- bow, tfidfのベクトル化(ipynb参照)

参考URL: http://www.madopro.net/entry/word2vec_with_tensorflow,
http://tensorflow.classcat.com/2016/03/12/tensorflow-cc-word2vec/

### gensimを使ったベクトル化プログラム(seed固定方法)

1. データセットに対してベクトル化のみ
2. workers = 1(並列処理しない)
3. シード値の固定(gensim内)
4. PYTHONHASHSEEDの固定

上記のパターンはgensim.models.word2vec参照([公式ドキュメント](https://radimrehurek.com/gensim/models/word2vec.html))

    seed (int) – Seed for the random number generator. Initial vectors for each word are seeded with a hash of the concatenation of word + str(seed). Note that for a fully deterministically-reproducible run, you must also limit the model to a single worker thread (workers=1), to eliminate ordering jitter from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires use of the PYTHONHASHSEED environment variable to control hash randomization).


`os.environ["PYTHONHASHSEED"] = "1"`

上記コードをプログラム内に記述することは無意味

`export PYTHONHASHSEED="1"`

上記コマンドを実行後、ファイルを実行することで学習が固定される

***

## やること

- bow, tfidf用のcalc_similarity(仮)の実装
- 前処理の実装(参照: https://qiita.com/Hironsan/items/2466fe0f344115aff177)
    - 正規表現確認サイト(https://regex101.com/)
    - 辞書データの拡充(金融用語)
- 過去実装分との違い, 操作マニュアルの作成
- 各種アルゴリズムのベクトル比較
    - 文章ベクトルについては, 類似度計算結果をcsvファイルで出力
    - 単語ベクトルの比較は検討中
- LSTMを使った文章生成プログラム
