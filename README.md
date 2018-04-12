# Word2Vec, Doc2Vec, fasttextのベクトル化プログラム作成

## 実装環境(実行環境)

python==3.5
tensorflow==1.3.0
gensim==3.4.0
fasttext==0.8.3


***

## 各種説明

### data

学習用データを保存

***

### ipynb

jupyter notebookを保存

***

### src

- nlp_with_gensim.py
- nlp_with_tensorflow.py
- nlp_with_fasttext.py
- weight.py
- module.py

***

### model

学習済みの重みを保存

***

## ファイル説明

### src/train.py

#### train

#### w2v

#### d2v

#### ft

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

参考URL: http://www.madopro.net/entry/word2vec_with_tensorflow,
http://tensorflow.classcat.com/2016/03/12/tensorflow-cc-word2vec/

### gensimを使ったベクトル化プログラム(seed固定方法)

1. データセットに対してベクトル化のみ
2. workersの固定(並列処理しない)
3. シード値の固定(word2vec内)
4. シード値の固定(numpy.random内)
5. PYTHONHASHSEEDの固定

上記のパターンはgensim.models.word2vec参照([公式ドキュメント](https://radimrehurek.com/gensim/models/word2vec.html))

    seed (int) – Seed for the random number generator. Initial vectors for each word are seeded with a hash of the concatenation of word + str(seed). Note that for a fully deterministically-reproducible run, you must also limit the model to a single worker thread (workers=1), to eliminate ordering jitter from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires use of the PYTHONHASHSEED environment variable to control hash randomization).


`os.environ["PYTHONHASHSEED"] = "1"`

上記コードをプログラム内に記述することは無意味

`export PYTHONHASHSEED="1"`

上記コマンドを実行後、ファイルを実行することで学習が固定される

***

## やること

- Word2Vec・Doc2Vec・FastTextでの単語ベクトルの学習(重み)の比較
- 過去に作成した機能の実装(各種学習方法 ← doc2vec)
- doc2vecの文書タグ付け
