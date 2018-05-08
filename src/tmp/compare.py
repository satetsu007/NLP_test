# coding: utf-8
from module import compare
import sys

def main():
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数

    # デバッグプリント
    if(argc != 3):   # 引数が足りない場合は、その旨を表示
        print('Usage: # python %s model_no word' % argvs[0])
        quit()         # プログラムの終了

    model_no = argvs[1]
    word = argvs[2]

    compare(model_no, word=word)

if __name__ == "__main__":
    main()