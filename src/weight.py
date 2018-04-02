import os
from gensim.models import word2vec, fasttext, doc2vec

def main():
    """
    単語ベクトルの確認
    """

    os.chdir("src")

    model_type = "fasttext"

    if model_type == "word2vec":
        model1 = word2vec.Word2Vec.load("../model/word2vec_0.model")
        model2 = word2vec.Word2Vec.load("../model/word2vec_1.model")

        print(model1.wv.word_vec("伊藤忠商事"))
        print(model2.wv.word_vec("伊藤忠商事"))

    elif model_type == "doc2vec":
        model1 = doc2vec.Doc2Vec.load("../model/doc2vec_0.model")
        model2 = doc2vec.Doc2Vec.load("../model/doc2vec_1.model")

        print(model1.wv.word_vec("伊藤忠商事"))
        print(model2.wv.word_vec("伊藤忠商事"))

    elif model_type == "fasttext":
        model1 = fasttext.FastText.load("../model/fasttext_0.model")
        model2 = fasttext.FastText.load("../model/fasttext_1.model")

        print(model1.wv.word_vec("伊藤忠商事"))
        print(model2.wv.word_vec("伊藤忠商事"))

if __name__ == "__main__":
    main()