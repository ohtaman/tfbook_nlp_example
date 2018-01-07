#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import csv
import os
import random
import sys
import zipfile

import numpy as np
import pandas as pd


DEFAULT_INDEX_FILE_PATH = 'data/raw/aozorabunko/index_pages/list_person_all_extended_utf8.zip'
DEFAULT_INPUT_DIR = 'data/parsed/aozorabunko/morph'
DEFAULT_OUTPUT_DIR = 'data/prepared/aozorabunko/classify/morph'
DEFAULT_VALID_SAMPLES = 1000
DEFAULT_TEST_SAMPLES = 1000

TRAIN_FILE_NAME = 'train.csv'
VALID_FILE_NAME = 'valid.csv'
TEST_FILE_NAME = 'test.csv'

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


def get_top10_authors(df):
    author_names = df.groupby('人物ID').apply(lambda x: max(x['姓'] + x['名']))
    counts = df['人物ID'].value_counts()[:10]

    top10 = pd.DataFrame(index=counts.index)
    top10['count'] = counts
    top10['name'] = top10.index.map(lambda x: author_names[x])
    return top10


def data_generator(paths, labels):
    for path, label in zip(paths, labels):
        with open(path) as i_:
            for line in i_:
                line = line.strip()
                if len(line) > 0:
                    yield line, label


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
        help='Path to the index file.'
    )
    parser.add_argument(
        '-i',
        '--input_dir',
        default=DEFAULT_INPUT_DIR,
        help='Path to the input directory. Default to {}'.format(DEFAULT_INPUT_DIR)
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        default=DEFAULT_OUTPUT_DIR,
        help='Path to the output directory. Default to {}.'.format(DEFAULT_OUTPUT_DIR)
    )
    parser.add_argument(
        '-v',
        '--valid_samples',
        type=int,
        default=DEFAULT_VALID_SAMPLES,
        help='Number of validation samples. Default to {}.'.format(DEFAULT_VALID_SAMPLES)
    )
    parser.add_argument(
        '-t',
        '--test_samples',
        type=int,
        default=DEFAULT_TEST_SAMPLES,
        help='Number of test samples. Default to {}'.format(DEFAULT_TEST_SAMPLES)
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=None,
        help='Use specific random seed.'
    )
    return parser


def main(argv):
    argparser = build_argparser()
    args = argparser.parse_args(argv)

    index_file = load_index_file(args.index_file)
    exists_only = extract_existing(index_file, args.input_dir)
    authors = get_top10_authors(exists_only)
    print('-- Top 10 authors --')
    print(authors)
    print('--------------------')

    top10_only = exists_only[exists_only['人物ID'].isin(authors.index)]

    df = pd.DataFrame(index=top10_only.index)
    df['path'] = top10_only['作品ID'].map(lambda x: get_content_path(x, args.input_dir))
    df['label'] = top10_only['人物ID'].apply(lambda x: authors.name[x])
    df = df.reindex([np.random.permutation(df.index)])

    train = df[:-400]
    valid = df[-400:-200]
    test = df[-200:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_output_path = os.path.join(args.output_dir, TRAIN_FILE_NAME)
    valid_output_path = os.path.join(args.output_dir, VALID_FILE_NAME)
    test_output_path = os.path.join(args.output_dir, TEST_FILE_NAME)

    random.seed(args.seed)
    save_data(data_generator(train.path, train.label), train_output_path)
    save_data(
        random.sample(
            list(data_generator(valid.path, valid.label)),
            args.valid_samples
        ),
        valid_output_path
    )
    save_data(
        random.sample(
            list(data_generator(test.path, test.label)),
            args.test_samples
        ),
        test_output_path
    )


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
