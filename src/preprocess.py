#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import unicodedata
import os
import re
import subprocess
import sys
import zipfile

import pandas as pd
import lxml.html

import MeCab


AOZORA_GIT_URL = 'https://github.com/aozorabunko/aozorabunko.git'
AOZORA_SITE_URL = 'http://www.aozora.gr.jp/'
INDEX_FILEPATH = 'index_pages/list_person_all_extended_utf8.zip'

DEFAULT_RAW_DIR = 'data/raw/aozorabunko'
DEFAULT_OUTPUT_DIR = 'data/parsed/aozorabunko/morph'


tagger = MeCab.Tagger()
tagger.parse('')  # バグ回避


def tokenize(text):
    node = tagger.parseToNode(text)
    while node is not None:
        feature = node.feature.split(',')
        if feature[0] != 'BOS/EOS':
            yield node.surface
        node = node.next


def parse_html(html_file):
    html = lxml.html.parse(html_file)
    # 本文を抽出
    main_text = html.find('//div[@class="main_text"]')
    if main_text is None:
        return None

    # ルビの除去
    for rt in main_text.findall('.//rt'):
        rt.getparent().remove(rt)
    for rp in main_text.findall('.//rp'):
        rp.getparent().remove(rp)

    # 注記と前後の不要な空白を除去
    text = re.sub(
        '［＃[^］]+］\n?',
        '',
        main_text.text_content(),
        flags=(re.MULTILINE)
    ).strip()

    # 正規化
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    return text


def get_avilable_contents(index_file):
    z = zipfile.ZipFile(index_file)
    with z.open(z.filelist[0].filename) as i_:
        df = pd.read_csv(i_)

    for _, row in df.iterrows():
        content_id = row['作品ID']
        url = row['XHTML/HTMLファイルURL']
        if not isinstance(url, str):
            continue
        if not url.startswith(AOZORA_SITE_URL):
            continue
        path = url[len(AOZORA_SITE_URL):]
        yield content_id, path


def download_data(data_dir):
    if not os.path.exists(data_dir):
        parent_dir = os.path.dirname(data_dir)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        ret = subprocess.call(
            'git clone {} {}'.format(AOZORA_GIT_URL, data_dir),
            shell=True
        )
        if ret != 0:
            raise RuntimeError('Failed to download dataset.')
        return True
    return False


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        '--raw_dir',
        default=DEFAULT_RAW_DIR,
        help='Path to the raw data directory. Defaults to {}'.format(DEFAULT_RAW_DIR)
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        default=DEFAULT_OUTPUT_DIR,
        help='Path to the output directory. Defaults to {}'.format(DEFAULT_OUTPUT_DIR)
    )
    parser.add_argument(
        '--no_morpheme',
        action='store_true',
        default=False,
        help='Disalbe morphological analysis.'
    )
    return parser


def main(argv):
    argparser = build_argparser()
    args = argparser.parse_args(argv)

    download_data(args.raw_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    # 青空文庫公式サイトにHTMLファイルが存在する作品についてのループ
    cnt = 0
    for content_id, rel_path in get_avilable_contents(os.path.join(
        args.raw_dir,
        INDEX_FILEPATH
    )):
        sys.stdout.write('\r\033[KProcessing {}\'th file...'.format(cnt))
        cnt += 1
        fname = '{}.txt'.format(content_id)
        output_path = os.path.join(args.output_dir, fname)
        raw_path = os.path.join(args.raw_dir, rel_path)
        parsed = parse_html(raw_path)
        if parsed:
            with open(output_path, 'w', encoding='utf-8') as o_:
                if args.no_morpheme:
                    print(parsed, file=o_)
                else:
                    for line in parsed.split('\n'):
                        print(' '.join(tokenize(line)), file=o_)
    print('Done.')

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
