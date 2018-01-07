#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils  # NOQA


DEFAULT_DATA_DIR = 'data/prepared/aozorabunko/classify/char'

VOCAB_SIZE = 2000
BATCH_SIZE = 128
MAX_LEN = 128
HIDDEN_SIZE = 128
EMB_SIZE = 128
STEPS = 10000
LOG_DIR = 'log'
MODEL_DIR = 'models/classify_char_rnn'

ENCODING = 'utf-8'
TRAIN_FILE_NAME = 'train.csv'
VALID_FILE_NAME = 'valid.csv'
TEST_FILE_NAME = 'test.csv'


def rnn_model(inputs, labels, rnn_cell, n_classes, emb_size, vocab_size, hidden_size):
    """
    RNNによる分類モデル

    Args:
      inputs (Tensor): [batch_size, num_steps] の入力データ.
      labels (Tensor): [batch_size] の正解ラベル.
      rnn_cell (RNNCell): RNNセル.
      n_classes (int): 分類クラス数.
      emb_size (int): 埋め込み空間のサイズ.
      vocab_size (int): 語彙数(文字の種類数).
      hidden_size (int): 全結合層のサイズ

    Returns:
      予測結果、確率スコア、誤差関数、正解率、学習演算子
    """
    # 単語の埋め込み
    with tf.variable_scope('emb'):
        emb_w = tf.get_variable(
            'w',
            shape=(vocab_size, emb_size),
            initializer=tf.contrib.layers.xavier_initializer()
        )
        emb = tf.nn.embedding_lookup(emb_w, inputs)

    # rnn_cell から RNN を構築
    with tf.variable_scope('rnn'):
        outputs, state = tf.nn.dynamic_rnn(rnn_cell, emb, dtype=tf.float32)
        last_outputs = outputs[:, -1]  # 分類問題では最後の出力のみを使う

    # RNN の出力から作者を推定する。ここでは2層のネットワークを使う
    with tf.variable_scope('output'):
        x = tf.layers.dense(last_outputs, hidden_size, activation=tf.nn.relu)
        logits = tf.layers.dense(x, n_classes)

    one_hot_labels = tf.one_hot(labels, n_classes)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits)
    )

    train_op = tf.train.AdamOptimizer().minimize(loss)
    softmax = tf.nn.softmax(logits)

    pred = tf.cast(tf.argmax(logits, axis=-1, name='pred'), tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))
    return pred, softmax, loss, acc, train_op


def build_datamart(df, labels, text_encoder, max_len):
    n_samples = len(df)
    x = np.zeros((n_samples, max_len), dtype=np.int32)
    for i, text in enumerate(df.text):
        encoded = text_encoder.encode(text)[:max_len]
        x[i][:len(encoded)] = encoded
    y = df.label.map(labels.index).astype(np.int32).values

    return x, y


def build_generator(x, y, batch_size):
    while True:
        n_samples = len(x)
        indices = np.random.permutation(range(n_samples))
        for i in range(0, n_samples, batch_size):
            batch_indices = [indices[(i + j) % n_samples] for j in range(batch_size)]
            batch_x = x[batch_indices]
            batch_y = y[batch_indices]
            yield batch_x, batch_y


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_dir',
        default=DEFAULT_DATA_DIR,
        help='Path to the data directory. Default to {}.'.format(DEFAULT_DATA_DIR)
    )
    return parser


def main(argv):
    argparser = build_argparser()
    args = argparser.parse_args(argv)

    train_file = os.path.join(args.data_dir, TRAIN_FILE_NAME)
    valid_file = os.path.join(args.data_dir, VALID_FILE_NAME)
    test_file = os.path.join(args.data_dir, TEST_FILE_NAME)

    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    test_df = pd.read_csv(test_file)

    label_names = sorted(train_df.label.unique())
    print('Labels: {}'.format(label_names))

    # 文字区切りとなるようtokenizerを指定
    text_encoder = utils.TextEncoder(tokenizer=utils.CharTokenizer())
    text_encoder.build_vocab(train_df.text, VOCAB_SIZE)

    train_x, train_y = build_datamart(train_df, label_names, text_encoder, MAX_LEN)
    valid_x, valid_y = build_datamart(valid_df, label_names, text_encoder, MAX_LEN)
    test_x, test_y = build_datamart(test_df, label_names, text_encoder, MAX_LEN)

    train_gen = build_generator(train_x, train_y, BATCH_SIZE)

    inputs = tf.placeholder(tf.int32, (None, MAX_LEN), name='inputs')
    labels = tf.placeholder(tf.int32, (None,), name='labels')

    rnn_cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
    pred, softmax, loss, acc, train_op = rnn_model(
        inputs,
        labels,
        rnn_cell,
        len(label_names),
        EMB_SIZE,
        text_encoder.vocab_size,
        HIDDEN_SIZE
    )

    checkpoint_path = os.path.join(MODEL_DIR, 'model.ckpt')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_sum_ = 0
        for step in range(1, STEPS + 1):
            # 学習用バッチを取得
            batch_x, batch_y = next(train_gen)
            loss_, _ = sess.run(
                [loss, train_op],
                feed_dict={
                    inputs: batch_x,
                    labels: batch_y
                }
            )
            loss_sum_ += loss_

            # 100ステップごとに検証用データで精度を確認
            if step % 100 == 0:
                pred_, valid_loss_, acc_ = sess.run(
                    [pred, loss, acc],
                    feed_dict={
                        inputs: valid_x,
                        labels: valid_y
                    }
                )
                # 学習用データについてのロスは過去100ステップの平均をとる
                avg_loss_ = loss_sum_/100
                loss_sum_ = 0
                print('step: {}, train loss: {}, valid loss: {}, acc: {}'.format(
                    step, avg_loss_, valid_loss_, acc_
                ))
                saver.save(sess, checkpoint_path, global_step=step)

    # Test
    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        pred_, loss_, acc_ = sess.run(
            [pred, loss, acc],
            feed_dict={
                inputs: test_x,
                labels: test_y
            }
        )
        print('loss: {}, acc: {}'.format(loss_, acc_))


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
