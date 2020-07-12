from operator import attrgetter
from collections import namedtuple
#from __future__ import absolute_import
#from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import nltk
import random
nltk.download("stopwords")

from warnings import warn
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy.linalg import svd as singular_value_decomposition
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfTransformer

from operator import attrgetter
from collections import namedtuple

import preparation


SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))

'''
class ItemsCount(object):
    def __init__(self, value):
        self._value = value

    def __call__(self, sequence):
        if isinstance(self._value, (bytes, str,)):
            if self._value.endswith("%"):
                total_count = len(sequence)
                percentage = int(self._value[:-1])
                # at least one sentence should be chosen
                count = max(1, total_count*percentage // 100)
                return sequence[:count]
            else:
                return sequence[:int(self._value)]
        elif isinstance(self._value, (int, float)):
            return sequence[:int(self._value)]
        else:
            ValueError("Unsuported value of items count '%s'." % self._value)

    def __repr__(self):
        return to_string("<ItemsCount: %r>" % self._value)
'''


class BaseSummarizer(object):
    
    def __call__(self, document, sentences_count):
        raise NotImplementedError("This method should be overriden in subclass")

    @staticmethod
    def normalize_word(word):
        return word.lower()

    @staticmethod
    def _get_best_sentences(sentences, count, rating):
        rate = rating

        infos = (SentenceInfo(s, o, rate(s))\
            for o, s in enumerate(sentences))

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        '''
        # get `count` first best rated sentences
        count = ItemsCount(count)
        infos = count(infos)
        # sort sentences by their order in document
        infos = sorted(infos, key=attrgetter("order"))
        '''
        
        infos = infos[:count]
        # sort sentences by their order in document
        infos = sorted(infos, key=attrgetter("order"))
        
        return tuple(i.sentence for i in infos)

class LsaSummarizer(BaseSummarizer):
    '''
    MIN_DIMENSIONS = 3
    REDUCTION_RATIO = 1/1
    '''

    _stop_words = list(stopwords.words('english'))

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = words

    def __call__(self, document, sentences_count):

        dictionary = self._create_dictionary(document)
        sentences = sent_tokenize(document)

        matrix = self._create_matrix(document, dictionary)
        matrix = self._compute_TfIdf(matrix)
        
        u, sigma, v = singular_value_decomposition(matrix, full_matrices=False)
        
        
        v = self._preprocess_matrix_V(v)
        ranks = iter(self._compute_ranks(v, sigma))
        
        return self._get_best_sentences(sentences, sentences_count,
            lambda s: next(ranks))

    def _create_dictionary(self, document):
        """Creates mapping key = word, value = row index"""

        words = word_tokenize(document)
        words = tuple(words)
        print(words)

        words = map(self.normalize_word, words)

        unique_words = sorted(frozenset(w for w in words if w not in self._stop_words))
        random.shuffle(unique_words)

        return dict((w, i) for i, w in enumerate(unique_words))

    def _create_matrix(self, document, dictionary):
        """
        contains number of occurences of words (rows) in senteces (cols).
        """
        document = str.lower(document)
        sentences = sent_tokenize(document)
        #print(sentences)
        words_count = len(dictionary)
        sentences_count = len(sentences)

        matrix = np.zeros((words_count, sentences_count))
        for col, sentence in enumerate(sentences):
            words = word_tokenize(sentence)
            for word in words:
                # only valid words is counted (not stop-words, ...)
                if word in dictionary:
                    row = dictionary[word]
                    matrix[row, col] += 1

        return matrix

    def _compute_TfIdf(self, old_matrix):
        
        tfidf = TfidfTransformer()
        matrix = tfidf.fit_transform(np.transpose(old_matrix)).toarray()

        return matrix

    def _preprocess_matrix_V(self, V):
        avg_per_concept = np.mean(V, axis = 0)
        return np.where(V > avg_per_concept, V, 0)

    def _compute_ranks(self, V, sigma):
        sentences_count = V.shape[1]
        ranks = np.zeros(sentences_count)
        for i in range(sentences_count):
            ranks[i] = np.sqrt(np.sum(V[:,i] * sigma))
        return ranks

class EnglishSummarizer(LsaSummarizer):

    def __init__(self):
      super().__init__()
      self.stop_words = stopwords.words("english")

class TextProcessor:

    def __init__(self, filepath, filetype, summarizer):
      self.prep = preparation.FilePreprocessor(filepath, filetype)
      self.summarizer = summarizer

    def __call__(self, splitText = False, l = 1, k = 5):
      text = self.prep.splitTextRandomly(l) if splitText else self.prep.joinText()
      summary = []
      for i in range(len(text)): 
        summary.append(" ".join(self.summarizer(text[i], k)))
      return summary

    '''
    def _compute_ranks_old(self, sigma, v_matrix):
        assert len(sigma) == v_matrix.shape[0]

        dimensions = max(LsaSummarizer.MIN_DIMENSIONS,
            int(len(sigma)*LsaSummarizer.REDUCTION_RATIO))
        powered_sigma = tuple(s**2 if i < dimensions else 0.0
            for i, s in enumerate(sigma))

        ranks = []
        
        for column_vector in v_matrix.T:
            rank = sum(s*v**2 for s, v in zip(powered_sigma, column_vector))
            ranks.append(math.sqrt(rank))

        return ranks
    '''
    