#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import csv
import os
import sys
import zipfile

import pandas as pd


DEFAULT_INDEX_FILE_PATH = 'data/raw/aozorabunko/index_pages/list_person_all_extended_utf8.zip'
DEFAULT_INPUT_DIR = 'data/parsed/aozorabunko/morph'
DEFAULT_OUTPUT_PATH = 'data/prepared/aozorabunko/generate/morph/train.csv'
DEFAULT_AUTHOR_NAME = '芥川竜之介'

ENCODING = 'utf-8'


def load_index_file(index_filepath):
    z = zipfile.ZipFile(index_filepath)
    with z.open(z.filelist[0]) as i_:
        return pd.read_csv(i_)


def extract_existing(df, content_dir):
    content_path = df['作品ID'].map(lambda x: get_content_path(x, content_dir))
    return df[content_path.map(os.path.exists)]


def get_content_path(content_id, content_dir):
    return os.path.join(content_dir, '{}.txt'.format(content_id))


def seq2seq_data_generator(paths):
    for path in paths:
        with open(path, encoding=ENCODING) as i_:
            prev_line = None
            for line in i_:
                line = line.strip()
                if prev_line and line:
                    yield prev_line, line
                prev_line = line


def rnn_data_generator(paths):
    for path in paths:
        with open(path, encoding=ENCODING) as i_:
            for line in i_:
                yield line


def save_data(generator, path):
    with open(path, 'w', encoding=ENCODING) as o_:
        writer = csv.DictWriter(o_, fieldnames=['text', 'label'])
        writer.writeheader()
        for x, y in generator:
            writer.writerow({'text': x, 'label': y})


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--index_file',
        default=DEFAULT_INDEX_FILE_PATH,
        help='Path to the index file. Default to {}.'.format(DEFAULT_INDEX_FILE_PATH)
    )
    parser.add_argument(
        '-i',
        '--input_dir',
        default=DEFAULT_INPUT_DIR,
        help='Path to the input directory. Default to {}.'.format(DEFAULT_INPUT_DIR)
    )
    parser.add_argument(
        '-o',
        '--output',
        default=DEFAULT_OUTPUT_PATH,
        help='Path to the output file. Default to {}.'.format(DEFAULT_OUTPUT_PATH)
    )
    parser.add_argument(
        '-a',
        '--author_name',
        default=DEFAULT_AUTHOR_NAME,
        help='Name of author to extract. Default to {}.'.format(DEFAULT_AUTHOR_NAME)
    )
    parser.add_argument(
        '-s',
        '--seq2seq',
        action='store_true',
        default=False,
        help='Process for seq2seq model.'
    )
    return parser


def main(argv):
    argparser = build_argparser()
    args = argparser.parse_args(argv)

    index_file = load_index_file(args.index_file)
    exists_only = extract_existing(index_file, args.input_dir)
    author_name = exists_only['姓'] + exists_only['名']
    authors_only = exists_only[author_name == args.author_name]
    paths = authors_only['作品ID'].map(lambda x: get_content_path(x, args.input_dir))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding=ENCODING) as o_:
        if args.seq2seq:
            generator = seq2seq_data_generator(paths)
            writer = csv.writer(o_)
            writer.writerow(['input', 'output'])
            for row in generator:
                writer.writerow(row)
        else:
            generator = rnn_data_generator(paths)
            for line in generator:
                o_.write(line)


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
