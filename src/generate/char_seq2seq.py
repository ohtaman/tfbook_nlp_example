#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils  # NOQA


DEFAULT_DATA_PATH = 'data/prepared/aozorabunko/generate/char/train.csv'

EMB_SIZE = 128
INPUT_VOCAB_SIZE = 2000
OUTPUT_VOCAB_SIZE = 2000
HIDDEN_SIZE = 256
N_LAYERS = 2
INPUT_MAX_LEN = 32
OUTPUT_MAX_LEN = 32
BATCH_SIZE = 64
STEPS = 100000
LOG_DIR = 'log'
MODEL_DIR = 'models/generate_char_seq2seq'


def encoder(inputs, sequence_length, rnn_cell, embedding_size, vocab_size):
    """
    エンコーダーの定義. エンコーダーの最終状態がデコーダーの初期状態となる

    Args:
        inputs　(Tensor): [batch_size, max_sequence_length] の入力データ.
        sequence_length (Tensor): [batch_size] で実際の入力文章の長さを表すデータ.
        rnn_cell (RNNCell): RNNセル
        embedding_size (int): 単語の埋め込み空間のサイズ.
        vocab_size (int): 語彙数(単語IDの最大値).
    Returns:
        エンコーダーの出力、エンコーダーの最終状態
    """
    with tf.variable_scope('encoder'):
        emb = tf.contrib.layers.embed_sequence(
            inputs,
            vocab_size=vocab_size,
            embed_dim=embedding_size
        )
        rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
            rnn_cell,
            emb,
            dtype=tf.float32,
            sequence_length=sequence_length
        )

    return rnn_outputs, rnn_state


def decoder(inputs, sequence_length, num_steps, rnn_cell, initial_state, embedding_size, vocab_size):
    """
    デコーダーの定義.

    Args:
        inputs (Tensor): [batch_size, max_sequence_length] の入力データ.
        sequence_length (Tensor): [batch_size] で実際の入力文章の長さを表すデータ.
        num_steps (int): 出力データの最大長.
        rnn_cell (RNNCell): RNNセル.
        intitial_state (Tensor or tupple of Tensor): RNNの初期状態(型はRNNセルに依存する).
        embedding_size (int): 単語の埋め込み空間のサイズ.
        vocab_size (int): 語彙数(単語IDの最大値).
    Returns:
        学習時の出力、推定時の出力
    """
    with tf.variable_scope('decoder'):
        with tf.variable_scope('emb'):
            emb_w = tf.get_variable(
                'w',
                shape=(vocab_size, embedding_size),
                initializer=tf.contrib.layers.xavier_initializer()
            )
            emb = tf.nn.embedding_lookup(emb_w, inputs)

    with tf.variable_scope('decoder'):
        output_layer = Dense(vocab_size)
        train_helper = tf.contrib.seq2seq.TrainingHelper(emb, sequence_length)
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

    with tf.variable_scope('decoder', reuse=True):
        start = inputs[:, 0]
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(emb_w, start, 0)
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


def seq2seq(encoder_inputs, encoder_len, encoder_cell, decoder_inputs, decoder_len, decoder_max_len, decoder_cell):
    """
    エンコーダ・デコーダモデル (Sequence to Sequence) の定義.

    Args:
        encoder_inputs (Tensor): [batch_size, max_sequence_length] のエンコーダ入力データ.
        encoder_len (Tensor): [batch_size] で実際のエンコーダ入力文章の長さを表すデータ.
        encoder_cell (RNNCell): エンコーダのcell
        decoder_inputs (Tensor): [batch_size, max_sequence_length] のデコーダ入力データ.
        decoder_len (Tensor): [batch_size] で実際のデコーダ入力文章の長さを表すデータ.
        decoder_max_len (Tensor): [batch_size] で実際のデコーダ入力文章の長さを表すデータ.
        decoder_cell (RNNCell): デコーダのcell.
    Returns:
        学習時の出力、推定時の出力
    """
    with tf.variable_scope('encoder'):
        encoder_outputs, encoder_last_state = encoder(
            encoder_inputs,
            encoder_len,
            encoder_cell,
            EMB_SIZE,
            INPUT_VOCAB_SIZE
        )
    with tf.variable_scope('decoder'):
        train_outputs, inference_outputs = decoder(
            decoder_inputs,
            decoder_len,
            decoder_max_len,
            decoder_cell,
            encoder_last_state,
            EMB_SIZE,
            OUTPUT_VOCAB_SIZE
        )
    return train_outputs, inference_outputs


def build_batch_generator(inputs, outputs, encoder_max_len, decoder_max_len, batch_size):
    BOS = utils.TextEncoder.RESERVED_TOKENS.index('<BOS>')
    EOS = utils.TextEncoder.RESERVED_TOKENS.index('<EOS>')

    n_samples = len(inputs)
    while True:
        encoder_inputs = np.zeros((batch_size, encoder_max_len), dtype=np.int32)
        encoder_len = np.zeros(batch_size, dtype=np.int32)
        decoder_inputs = np.zeros((batch_size, decoder_max_len), dtype=np.int32)
        decoder_len = np.zeros(batch_size, dtype=np.int32)
        decoder_outputs = np.zeros((batch_size, decoder_max_len), dtype=np.int32)
        for e in range(batch_size):
            idx = np.random.randint(n_samples - 1)
            encoder_input = inputs[idx][-(encoder_max_len - 1):] + [EOS]
            encoder_len[e] = len(encoder_input)
            encoder_inputs[e][:encoder_len[e]] = encoder_input

            decoder_output = outputs[idx][:decoder_max_len]
            if len(decoder_output) < decoder_max_len:
                decoder_output += [EOS]
            decoder_len[e] = len(decoder_output)
            decoder_outputs[e][:decoder_len[e]] = decoder_output
            decoder_inputs[e][0] = BOS
            decoder_inputs[e][1:] = decoder_outputs[e][:-1]
        yield encoder_inputs, encoder_len, decoder_inputs, decoder_outputs, decoder_len

def build_test_data(inputs, encoder_max_len):
    BOS = utils.TextEncoder.RESERVED_TOKENS.index('<BOS>')
    EOS = utils.TextEncoder.RESERVED_TOKENS.index('<EOS>')

    n_samples = len(inputs)
    encoder_inputs = np.zeros((n_samples, encoder_max_len), dtype=np.int32)
    encoder_len = np.zeros(n_samples, dtype=np.int32)
    for idx in range(n_samples):
        encoder_input = inputs[idx][-(encoder_max_len - 1):] + [EOS]
        encoder_len[idx] = len(encoder_input)
        encoder_inputs[idx][:encoder_len[idx]] = encoder_input
    return encoder_inputs, encoder_len


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--no_train',
        action='store_true',
        default=False,
        help='Do not train.'
    )
    parser.add_argument(
        '--train_data',
        default=DEFAULT_DATA_PATH,
        help='Path to the train data. Default to {}.'.format(DEFAULT_DATA_PATH)
    )
    parser.add_argument(
        '--test_data',
        default=None,
        help='Path to the test data.'
    )
    return parser


def main(argv):
    argparser = build_argparser()
    args = argparser.parse_args(argv)
    train_df = pd.read_csv(args.train_data)

    input_text_encoder = utils.TextEncoder(tokenizer=utils.CharTokenizer())
    input_text_encoder.build_vocab(train_df.input, INPUT_VOCAB_SIZE)
    output_text_encoder = utils.TextEncoder(tokenizer=utils.CharTokenizer())
    output_text_encoder.build_vocab(train_df.output, OUTPUT_VOCAB_SIZE)
    train_gen = build_batch_generator(
        train_df.input.map(input_text_encoder.encode),
        train_df.output.map(output_text_encoder.encode),
        INPUT_MAX_LEN,
        OUTPUT_MAX_LEN,
        BATCH_SIZE
    )

    if args.test_data:
        test_df = pd.read_csv(args.test_data)
        test_inputs_, test_len_ = build_test_data(
            test_df.input.map(input_text_encoder.encode),
            INPUT_MAX_LEN,
        )

    encoder_cell = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
        for _ in range(N_LAYERS)
    ])

    decoder_cell = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
        for _ in range(N_LAYERS)
    ])

    encoder_inputs = tf.placeholder(tf.int32, (None, None), name='encoder_inputs')
    encoder_len = tf.placeholder(tf.int32, (None,), name='encoder_len')
    decoder_inputs = tf.placeholder(tf.int32, (None, None), name='decoder_inputs')
    decoder_len = tf.placeholder(tf.int32, (None,), name='decoder_len')
    targets = tf.placeholder(tf.int32, (None, None), name='targets')

    train_outputs, inference_outputs = seq2seq(
        encoder_inputs,
        encoder_len,
        encoder_cell,
        decoder_inputs,
        decoder_len,
        OUTPUT_MAX_LEN,
        decoder_cell
    )

    with tf.variable_scope('loss'):
        masks = tf.sequence_mask(decoder_len, OUTPUT_MAX_LEN, dtype=tf.float32, name='masks')
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=train_outputs.rnn_output,
            targets=targets,
            weights=masks
        )
        tf.summary.scalar('train_loss', loss)
    with tf.variable_scope('opt'):
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

    checkpoint_path = os.path.join(MODEL_DIR, 'model.ckpt')
    saver = tf.train.Saver()
    if not args.no_train:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
            if ckpt:
                print('Use checkpoint file: ' + ckpt.model_checkpoint_path)
                saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
            sum_loss = 0
            for step in range(1, STEPS + 1):
                encoder_inputs_, encoder_len_, decoder_inputs_, decoder_outputs_, decoder_len_ = next(train_gen)
                loss_, _ = sess.run(
                    [loss, train_op],
                    feed_dict={
                        encoder_inputs: encoder_inputs_,
                        encoder_len: encoder_len_,
                        decoder_inputs: decoder_inputs_,
                        decoder_len: decoder_len_,
                        targets:  decoder_outputs_
                    }
                )
                sum_loss += loss_
                if step%100 == 0:
                    saver.save(sess, checkpoint_path, global_step=step)

                    print(step, sum_loss/100)
                    sum_loss = 0
                    # 直前の input_ に対する推定結果を例示
                    train_inferences = sess.run(
                        inference_outputs.sample_id,
                        feed_dict={
                            encoder_inputs: encoder_inputs_,
                            encoder_len: encoder_len_,
                            decoder_inputs: [[utils.TextEncoder.RESERVED_TOKENS.index('<BOS>')]]*len(encoder_inputs_)
                        }
                    )
                    for _ in range(3):
                        print('---- INPUT (TRAIN) ----')
                        print(input_text_encoder.decode(encoder_inputs_[_]))
                        print('---- OUTPUT ----')
                        print(output_text_encoder.decode(train_inferences[_]))

                    # テストデータについてもはじめの3件だけ結果を見る
                    if args.test_data:
                        test_inferences = sess.run(
                            inference_outputs.sample_id,
                            feed_dict={
                                encoder_inputs: test_inputs_[:3],
                                encoder_len: test_len_[:3],
                                decoder_inputs: [[utils.TextEncoder.RESERVED_TOKENS.index('<BOS>')]]*len(test_inputs_[:3])
                            }
                        )
                        for _ in range(len(test_inferences_)):
                            print('---- INPUT (TEST) ----')
                            print(input_text_encoder.decode(test_inputs_[_]))
                            print('---- OUTPUT ----')
                            print(output_text_encoder.decode(test_inferences[_]))
    if args.test_data:
        ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        print('Use checkpoint file: ' + ckpt.model_checkpoint_path)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
            inferences = sess.run(
                inference_outputs.sample_id,
                feed_dict={
                    encoder_inputs: test_inputs_,
                    encoder_len: test_len_,
                    decoder_inputs: [[utils.TextEncoder.RESERVED_TOKENS.index('<BOS>')]]*len(test_inputs_)
                }
            )
            for _ in range(len(test_inputs_)):
                print('---- INPUT ----')
                print(input_text_encoder.decode(test_inputs_[_]))
                print('---- OUTPUT ----')
                print(output_text_encoder.decode(inferences[_]))


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
