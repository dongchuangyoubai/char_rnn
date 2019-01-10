import numpy as np
import copy
import time
import tensorflow as tf
import pickle
import codecs

class TextConverter(object):
    def __init__(self, text, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            # vocab = set(text)
            vocab_count = {}
            for i in text:
                if i not in vocab_count:
                    vocab_count[i] = 0
                vocab_count[i] += 1
            vocab = vocab_count.keys()
            # print(vocab_count)
            # print(len(vocab))
            vocab_count = sorted(vocab_count.items(), key=lambda d: d[1], reverse=True)
            # print(vocab_count)
            self.vocab = [x[0] for x in vocab_count]
            # print(self.vocab)

        self.char2int = {v: k for k, v in enumerate(self.vocab)}
        self.int2char = dict(enumerate(self.vocab))
        # print(self.char2int)
        # print(self.int2char)

    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string('input_file', 'data/shakespeare.txt', 'utf8 encoded text file')
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    t = TextConverter(text)
