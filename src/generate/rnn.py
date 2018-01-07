#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils  # NOQA


DEFAULT_DATA_PATH = 'data/prepared/aozorabunko/generate/morph/train.csv'

EMB_SIZE = 128
VOCAB_SIZE = 50000
HIDDEN_SIZE = 256
N_LAYERS = 2
BATCH_SIZE = 64
MAX_LEN = 32
STEPS = 10000
LOG_DIR = 'log'
MODEL_DIR = 'models/generate_rnn'

ENCODING = 'utf-8'


def rnn_model(inputs, num_steps, rnn_cell, initial_state, embedding_size, vocab_size):
    """
    RNNによる文章生成モデル.学習時と推定時でネットワーク構造が異なる。

    Args:
        inputs (Tensor): [batch_size, max_sequence_length] の入力データ.
        num_steps (int): 出力データの最大長.
        rnn_cell (RNNCell): RNNセル.
        intitial_state (Tensor or tupple of Tensor): RNNの初期状態(型はRNNセルに依存する).
        embedding_size (int): 単語の埋め込み空間のサイズ.
        vocab_size (int): 語彙数(単語IDの最大値).
    Returns:
        学習時の出力、推定時の出力
    """
    with tf.variable_scope('decoder'):
        # 分類器と同様に単語の埋め込みをおこなう
        with tf.variable_scope('emb'):
            emb_w = tf.get_variable(
                'w',
                shape=(vocab_size, embedding_size),
                initializer=tf.contrib.layers.xavier_initializer()
            )
            emb = tf.nn.embedding_lookup(emb_w, inputs)

    # 学習用のネットワーク定義
    with tf.variable_scope('decoder'):
        # seq2seq.Decoder では、 output_layerを経由して入力値を調整する
        output_layer = Dense(vocab_size)
        # 学習時は TrainingHelper を用いる
        train_helper = tf.contrib.seq2seq.TrainingHelper(
            emb,
            tf.ones_like(inputs[:, 0])*num_steps
        )
        train_decoder = tf.contrib.seq2seq.BasicDecoder(
            rnn_cell,
            train_helper,
            initial_state,
            output_layer
        )
        train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            train_decoder,
            impute_finished=True,
            maximum_iterations=num_steps
        )

    # 推定(文章生成)用のネットワーク定義
    with tf.variable_scope('decoder', reuse=True):
        # Helperを選ぶことで「次の単語として何を使うか」が決まる。
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            emb_w,
            inputs[:, 0],
            0
        )
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            rnn_cell,
            inference_helper,
            initial_state,
            output_layer
        )
        inference_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder,
            impute_finished=True,
            maximum_iterations=num_steps
        )

    return train_outputs, inference_outputs


def build_batch_generator(text, batch_size, max_len):
    n_tokens = len(text)
    while True:
        inputs = np.zeros((batch_size, max_len), dtype=np.int32)
        outputs = np.zeros((batch_size, max_len), dtype=np.int32)
        for e in range(batch_size):
            idx = np.random.randint(n_tokens - max_len)
            inputs[e] = text[idx: idx + max_len]
            outputs[e] = text[idx + 1: idx + max_len + 1]
        yield inputs, outputs


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data',
        default=DEFAULT_DATA_PATH,
        help='Path to the data directory.'
    )
    return parser


def main(argv):
    argparser = build_argparser()
    args = argparser.parse_args(argv)

    with open(args.data, encoding=ENCODING) as i_:
        text = i_.read()

    text_encoder = utils.TextEncoder()
    text_encoder.build_vocab([text], VOCAB_SIZE)
    gen = build_batch_generator(
        text_encoder.encode(text),
        BATCH_SIZE,
        MAX_LEN
    )

    inputs = tf.placeholder(tf.int32, (None, None), name='inputs')
    targets = tf.placeholder(tf.int32, (None, None), name='targets')

    rnn_cell = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
        for _ in range(N_LAYERS)
    ])
    initial_state = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    train_outputs, inference_outputs = rnn_model(
        inputs,
        MAX_LEN,
        rnn_cell,
        initial_state,
        EMB_SIZE,
        VOCAB_SIZE
    )

    with tf.variable_scope('loss'):
        masks = tf.ones_like(targets, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            train_outputs.rnn_output,
            targets,
            masks
        )

    with tf.variable_scope('opt'):
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

    checkpoint_path = os.path.join(MODEL_DIR, 'model.ckpt')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sum_loss = 0
        for step in range(1, STEPS + 1):
            inputs_, outputs_ = next(gen)
            loss_, _ = sess.run(
                [loss, train_op],
                feed_dict={
                    inputs: inputs_,
                    targets: outputs_
                }
            )
            sum_loss += loss_
            if step % 100 == 0:
                saver.save(sess, checkpoint_path, global_step=step)

                print(step, sum_loss/100)
                sum_loss = 0
                # 直前の input_ に対する推定結果を例示
                inferences = sess.run(
                    inference_outputs.sample_id,
                    feed_dict={
                        inputs: inputs_[:, :1]
                    }
                )

                for _ in range(3):
                    print('---- SEED ----')
                    print(text_encoder.decode(inputs_[_, :1]))
                    print('---- OUTPUT ----')
                    print(text_encoder.decode(inferences[_]))


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
