# サンプルコードについて

## 環境構築

Python3 及び MeCab がインストールされていることを前提としています。
追加のPythonパッケージは`requirements.txt`もしくは`requirements-nogpu.txt`にまとめてあり、

```bash
$ pip install -r requirements.txt
```

もしくは

```bash
$ pip install -r requirements-nogpu.txt
```

でインストール可能です。

## 4-2 データの準備と前処理

**4-2 データの準備と前処理** で取り扱った青空文庫の前処理スクリプトは `src/preprocess.py` にまとめてあります。
以下のコマンドを実行すると、形態素解析を含む前処理の実行結果はデフォルトで `data/parsed/aozorabunko/morph` に出力されます。青空文庫のデータが存在しない場合はダウンロードしてくれますが、非常に時間がかかります。

```bash
$ python src/preprocess.py
```

文字ベースモデルのために形態素解析をしない場合は `--no_morpheme` オプションをつけ、 `-o` で出力先を設定します。

```bash
$ python src/preprocess.py --no_morpheme -o data/parsed/aozorabunko/char
```

## 4-3 文書を分類してみよう

**4-3 文書を分類してみよう** で取り扱ったスクリプトは `src/classify` 以下にまとめてあります。
実際に分類に進む前に `src/classify/preprocess.py` によってデータの整形と分割(学習用/検証用/テスト用)を行います。
結果はデフォルトで `data/prepared/aozorabunko/classify/` 以下に、 text, label というヘッダ付きのCSVファイルに出力されます。

```bash
$ python src/classify/preprocess.py
```

分類は以下のコマンドで実行できます。

```bash
$ python src/classify/rnn.py
```

文字ベースのモデルの場合は、以下のようになります。

```bash
$ python src/classify/preprocess.py -i data/parsed/aozorabunko/char -o data/prepared/aozorabunko/classify/char
$ python src/classify/char_rnn.py -d data/prepared/aozorabunko/classify/char
```

## 4-4 生成モデルで遊ぼう
**4-4 生成モデルで遊ぼう** 取り扱ったスクリプトは `src/generate` 以下にまとめてあります。
前処理は `src/generate/preprocess.py` で実行できます。
RNNモデルと sequence to sequence モデルでデータ形式が違うため、 sequence to sequence モデル用のデータを作成するには `--seq2seq` オプションを指定する必要があります。出力結果を見ていただければわかりますが、RNNモデル用のデータは単なるテキスト、sequence to sequence モデル用のデータは2列のCSVです。

```bash
$ python src/generate/preprocess.py  # RNNモデルを利用する場合
$ python src/generate/preprocess.py --seq2seq  # seq2seq モデルを利用する場合
```

文字ベースのモデルを利用する場合は、 `-i` と `-o` オプションで先ほどのディレクトリを指定します。

```bash
$ python src/generate/preprocess.py [--seq2seq] -i data/parsed/aozorabunko/char -o data/prepared/aozorabunko/generate/char/train.csv
```

学習と文章生成は `rnn.py`、`seq2seq.py`、`char_seq2seq.py` で実行できます。

```bash
$ python src/generate/rnn.py  # RNN を試す場合
$ python src/generate/seq2seq.py  # sequence to sequence を試す場合
$ python src/generate/char_seq2seq.py  # char based sequence to sequence を試す場合
```

また、私の不手際で、書籍の **表4-8** および **表4-9** には、学習データに対する出力結果を載せてしまいました。
（正誤表にて、その旨を明記する予定です。）
実際には学習に用いていないデータで評価した結果を見る必要があります。

参考までに、単語ベースのseq2seqモデルを使って、芥川の文ではなく、太宰治の文章と適当にピックアップして入力した場合の出力を以下に示します。
「ロマンス」という単語に反応してか、「空模様」といった出力が得られたり、会話文（「で始まり、」で終わる文）には、会話文を出力しようとしているのがわかるかと思います。

|input|output|
|:---:|:----:|
|&lt;UNK&gt; 、 五 人 あっ て 、 みんな ロマンス が 好き だっ た 。|その間 に 空模様 が 変っ た 。 対岸 を 塞い だ 山 の 空 に は 、 いつか 二 度 か 、 まっ 花 と 云う もの が 、 確か ながら 、|
|「 王様 は 、 人 を 殺し ます 。 」|「 は 、 御 二 匹 の よう な 思い です が 、 それでも あの 内 なら ば 、 い ぬ 事 じゃ あり ませ ん が ござい まし た 。 」|
|私 は 、 その 男 の 写真 を 三 葉 、 見 た こと が ある 。|おれ の 行く手 に は どこ か で ある が 、 その 暗黒 の 硝子 戸 か の 間 に 、 何 か 云 ふ 所 から 来る 。 人 の 高い|
