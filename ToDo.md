# Word2Vec, Doc2Vecのベクトル化プログラム作成&シード値固定による再現性の確認作業


## やったこと

- データを整理
- テキストファイルとデータフレームを結合
- テキストファイルを分かち書き
- テキストファイルを結合(light, middle, allの3パターン)

Word2Vec(gensim)を使ったベクトル化プログラム

1. データセットに対してベクトル化のみ
2. workersの固定(並列処理しない)
3. シード値の固定(word2vec内)
4. シード値の固定(numpy.random内)
5. PYTHONHASHSEEDの固定

上記のパターンはgensim.models.word2vec参照([公式ドキュメント](https://radimrehurek.com/gensim/models/word2vec.html))

***

## やること

- Word2Vecで作成したプログラムをDoc2Vecでも同様に作成する
- Word2VecとDoc2Vecでの単語ベクトルの学習(重み)の比較
- gensimを使用しないWord2Vecの作成(tensorflow使用)

参考URL: http://www.madopro.net/entry/word2vec_with_tensorflow,
http://tensorflow.classcat.com/2016/03/12/tensorflow-cc-word2vec/

***

### 読むべきもの

gensim: https://radimrehurek.com/gensim/<br>
tensorflow: https://www.tensorflow.org/<br>
