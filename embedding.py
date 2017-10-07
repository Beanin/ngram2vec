"""Framework to transform letter ngrams into k-dimensional vector
   using CBOW model applied to letter ngrams
   https://iksinc.wordpress.com/tag/continuous-bag-of-words-cbow
"""

import math
import numpy as np
import string
import tensorflow as tf
import re

from collections import Counter

ALPHABET_LEN = len(string.ascii_lowercase)
MAIN_TF_DEVICE = '/cpu:0'

UNK = 0
UNK_STR = "UNK"


class Base26(object):
    """ Transformation between integers and
        strings of english lowercase letters
    """
    @staticmethod
    def decode(seq):
        base = ALPHABET_LEN
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

    def batches(self, batch_size):
        """
            generates batches of cbow batch_size
            from text passed as a list of base64-decoded
            ngrams
        """
        while True:
            batch = np.ndarray((batch_size, 2 * self._window_size),
                               dtype=np.int32)
            labels = np.ndarray((batch_size, 1), dtype=np.int32)
            for i in range(batch_size):
                batch[i], labels[i] = self.next_batch()
                self._update_cursor()

            yield batch, labels


class NGRAMDictionary(object):
    """ Contain vocabulary of ngrams and words
        and required mappings between them:
        - enumerations
        - one-hot-encoding of words as a set of ngrams
    """

    def __init__(self, data, ngram_range=(2, 2), vocabulary_size=50000,
                 token_pattern="[A-Z\-\']{2,}(?![a-z])|[A-Z\-\']"
                 "[a-z\-\']+(?=[A-Z])|[\'\w\-]+",
                 tokenizer=None):
        """ Initialize text and vocabularies with respect to parameters
            params:
                data - text to process
                type: string

                ngram_range - inclusive range of ngrams to consider
                type: 2D-tuple

                vocabulary_size - number of words to keep
                type: int

                tokenizer - function to transorm string into list of tokens
                type: callable

                token_pattern - regexp pattern to tokenize text on words
                                regexp tokenization is used if tokenizer
                                is not specified
                type: string
        """
        self._ngram_range = ngram_range
        self._vocabulary_size = vocabulary_size

        # init tokenizer
        if tokenizer is None:
            p = re.compile(token_pattern)
            self._tokenizer = lambda s: p.findall(s)
        else:
            self._tokenizer = tokenizer

        # tokenize text
        words = [token.lower() for token in self._tokenizer(data)]

        # save most common words
        self._vocabulary = [UNK_STR] + \
                           [w for w, _ in
                            Counter(words).most_common(self._vocabulary_size)]
        self._reverse_vocabulary = {w: i for i, w in
                                    enumerate(self._vocabulary)}

        # extract ngrams
        ngrams = set()
        for ngram_sz in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(data) - ngram_sz + 1):
                ngrams.add(data[i: i + ngram_sz + 1])

        self._ngrams = [UNK_STR] + list(ngrams)
        self._reverse_ngrams = {w: i for i, w in
                                enumerate(self._ngrams)}

    def enumerate_ngram(self, ngrams):
        return [self._reverse_ngrams.get(t, UNK) for t in ngrams]

    def decode_ngram(self, indexes):
        return [self._ngrams[i] for i in indexes]

    def _one_hot_encode1(self, word):
        ngrams = []
        for ngram_sz in range(self._ngram_range[0], self._ngram_range[1] + 1):
            for i in range(len(word) - ngram_sz + 1):
                ngrams.append(word[i: i + ngram_sz + 1])

        cnts = Counter(ngrams)
        return np.array([cnts[i] for i in self._ngrams])

    def one_hot_encode(self, words):
        return np.array([self._one_hot_encode1(w) for w in words])


class NGRAMVectorizer(object):
    def __init__(self, N=2, dim=64, batch_size=64, window_size=1,
                 sampled=0.1, tol=0.001):
        """ Initialize new NGRAMVectorizer instance and all its parameters
            params:
                N - ngram length
                type: int

                dim - embeddings size
                type: int

                batch_size - size of training batches
                type: int

                window_size - how many neighbour words should
                              be considered in CBOW
                type: int

                sampled - relation of the number of samples
                          to  the whole vocabulary size
                type: float from (0, 1)

                tol - threshold of the loss function
                      change, when to stop SGD
                type: float
        """

        self._N = N
        self._dim = dim
        self._text = None
        self._batch_size = batch_size
        self._window_size = window_size
        self._sampled = sampled
        self._tol = tol

        self._embeddings = np.random.uniform(-1.0, 1.0,
                                             [self.vocabulary_size, self._dim])

    @property
    def vocabulary_size(self):
        return ALPHABET_LEN ** self._N

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

    def _text2list(self, text):
        text_len = len(text)
        return [self._ngram2int(text[i * self._N: (i + 1) * self._N])
                for i in range(text_len // self._N)]

    def _list2text(self, list):
        return "".join(map(self._int2ngram, list))

    def fit(self, text):
        """ Given the text, train embeddings using CBOW method

        parameters:
        text - text to train
        type: string

        returns:
            None
        """
        data = self._text2list(text)

        graph = tf.Graph()
        with graph.as_default():
            with graph.device(MAIN_TF_DEVICE):
                train_input = tf.placeholder(tf.int32,
                                             [self._batch_size,
                                              self._window_size * 2])
                train_labels = tf.placeholder(tf.int32, [self._batch_size, 1])

                embeddings = tf.Variable(
                    tf.constant(self._embeddings, dtype=tf.float32)
                )

                softmax_weights = tf.Variable(tf.truncated_normal(
                    [self.vocabulary_size, self._dim],
                    stddev=1 / math.sqrt(self._dim)))
                softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

                emb = tf.nn.embedding_lookup(embeddings, train_input)
                features = tf.reduce_mean(emb, 1)

                loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                    weights=softmax_weights,
                    biases=softmax_biases,
                    labels=train_labels,
                    inputs=features,
                    num_classes=self.vocabulary_size,
                    num_sampled=int(self._sampled * self.vocabulary_size)
                ))

                optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),
                                             1, keep_dims=True))
                normalized_embeddings = embeddings / norm

        batch_gen = BatchGenerator(data, self._window_size)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            average_loss, prev_loss, stop_counter = 0, -1, 0
            step = 1
            for X, labels in batch_gen.batches(self._batch_size):
                feed_dict = {train_input: X, train_labels: labels}
                _, loss_val = session.run(
                    [optimizer, loss],
                    feed_dict=feed_dict)

                average_loss += loss_val

                if step % 1000 == 0:
                    average_loss = average_loss / 1000
                    print("Average loss %f at step %d"
                          % (average_loss, step))

                    if prev_loss - average_loss < self._tol:
                        stop_counter += 1
                        if stop_counter > 10:
                            break
                    else:
                        stop_counter = 0

                    average_loss, prev_loss = 0, average_loss

                step += 1

            self._embeddings = normalized_embeddings.eval()

    def transform(self, text):
        """ Transform text into sequence of vectors
            based on ngrams

            parameters:
            text  - text to extract ngrams
            type: string

            returns:
                np.ndarray containing sequence of ngrams
        """
        return self._embeddings[(self._text2list(text), )]
