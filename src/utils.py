# -*- coding:utf-8 -*-

import abc
import collections


class Tokenizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode(self, text):
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, tokens):
        raise NotImplementedError()


class WordTokenizer(Tokenizer):
    def __init__(self, delimiter=' '):
        self.delimiter = delimiter

    def encode(self, text):
        return text.split(self.delimiter)

    def decode(self, tokens):
        return self.delimiter.join(tokens)


class CharTokenizer(Tokenizer):
    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return ''.join(tokens)


class MeCabTokenizer(Tokenizer):
    def __init__(self, mecab_tagger=None):
        import MeCab

        self.mecab_tagger = mecab_tagger if mecab_tagger else MeCab.Tagger()
        # Hack. see http://d.hatena.ne.jp/oggata/20161206
        self.mecab_tagger.parse('')

    def encode(self, text):
        node = self.mecab_tagger.parseToNode(text)
        while node is not None:
            feature = node.feature.split(',')
            if feature[0] != 'BOS/EOS':
                yield node.surface
            node = node.next

    def decode(self, tokens):
        return ''.join(tokens)


class TextEncoder(object):
    RESERVED_TOKENS = [
        '<PAD>',
        '<BOS>',
        '<EOS>',
        '<UNK>'
    ]

    def __init__(self, vocab=None, tokenizer=WordTokenizer()):
        if vocab is not None:
            self.vocab = vocab
            self.vocab_reversed = self._reverse_vocab(self.vocab)
        else:
            self.vocab = None
            self.vocab_reversed = None
        self.num_reserved_tokens = len(self.RESERVED_TOKENS)
        self.tokenizer = tokenizer

    @property
    def vocab_size(self):
        if self.vocab is not None:
            return len(self.RESERVED_TOKENS) + len(self.vocab)
        else:
            return len(self.RESERVED_TOKENS)

    @staticmethod
    def _reverse_vocab(vocab):
        return {token: id_ for id_, token in enumerate(vocab)}

    def build_vocab(self, texts, vocab_size=50000):
        counter = collections.Counter()
        for text in texts:
            counter.update(self.tokenizer.encode(text))
        len_vocab = vocab_size - len(self.RESERVED_TOKENS)
        self.counter = counter
        self.word_count = sum(counter.values())
        self.vocab = [token for token, count in sorted(
            counter.most_common(len_vocab),
            key=lambda x: (-x[1], x[0])
        )]
        self.vocab_reversed = self._reverse_vocab(self.vocab)

    def encode(self, text):
        encoded = []
        for token in self.tokenizer.encode(text):
            if token in self.vocab_reversed:
                encoded.append(self.vocab_reversed[token] + self.num_reserved_tokens)
            else:
                encoded.append(self.RESERVED_TOKENS.index('<UNK>'))
        return encoded

    def decode(self, ids):
        tokens = (self._decode_a_token(id_) for id_ in ids)
        return self.tokenizer.decode(tokens)

    def _decode_a_token(self, token_id):
        if token_id < self.num_reserved_tokens:
            token = self.RESERVED_TOKENS[token_id]
            if token != '<UNK>':
                token = ''
        else:
            token = self.vocab[token_id - self.num_reserved_tokens]
        return token
