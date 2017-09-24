"""Framework to transform letter ngrams into k-dimensional vector
   using CBOW model applied to letter ngrams
   https://iksinc.wordpress.com/tag/continuous-bag-of-words-cbow
"""

import numpy as np
import string


class Base26(object):
    """ Transformation between integers and
        strings of english lowercase letters
    """
    @staticmethod
    def decode(seq):
        base = len(string.ascii_lowercase)
        res = 0
        mult = 1
        for ch in seq:
            if ord(ch) < ord('a') or ord(ch) > ord('z'):
                raise ValueError("Invalid character %s. "
                                 "It should be ascii lowercase letter")

            res += mult * (ord(ch) - ord('a'))
            mult *= base

        return res

    @staticmethod
    def encode(x):
        base = len(string.ascii_lowercase)
        res = []

        while x > 0:
            res.append(chr(x % base + ord('a')))
            x //= base

        return "".join(res)


class BatchGenerator(object):
    def __init__(self, data, window_size):
        self._data = data
        self._window_size = window_size

        self._data_sz = len(self._data)
        if self._data_sz < 2 * window_size + 1:
            raise ValueError("Data is too small")

        self._cursor = window_size
        self._max_cursor = self._data_sz - self._window_size - 1

    def next_batch(self):
        return (np.concatenate([
            self._data[self._cursor - self._window_size: self._cursor],
            self._data[self._cursor + 1: self._cursor + self._window_size + 1]
        ], 0), self._data[self._cursor])

    def _update_cursor(self):
        self._cursor = max(
            self._window_size,
            (self._cursor + 1) % self._max_cursor
        )

    def batches(self, batch_size, count):
        """
            generates batches of cbow batch_size
            from text passed as a list of base64-decoded
            ngrams
        """
        for _ in range(count):
            batch = np.ndarray((batch_size, 2 * self._window_size),
                               dtype=np.int32)
            labels = np.ndarray((batch_size, 1), dtype=np.int32)
            for i in range(batch_size):
                batch[i], labels[i] = self.next_batch()
                self._update_cursor()

            yield batch, labels


class NGRAMVectorizer(object):
    def __init__(self, N=2, dim=64):
        self._N = N
        self._dim = dim
        self._text = None

    def _ngram2int(self, seq):
        if len(seq) != self._N:
            raise ValueError(
                "Invalid ngram len: %d, expected: %d" % (len(seq), self._N))

        return Base26.decode(seq)

    def _int2ngram(self, x):
        enc = Base26.encode(x)
        if len(enc) > self._N:
            raise ValueError("Invalid ngram number:%d" % x)

        return "a" * (self._N - len(enc)) + enc

    def text2list(self, text):
        text_len = len(text)
        return [self._ngram2int(text[i * self._N: (i + 1) * self._N])
                for i in range(text_len // self._N)]

    def list2text(self, list):
        return "".join(map(self._int2ngram, list))
