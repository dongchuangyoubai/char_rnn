import numpy as np
import copy
import time
import tensorflow as tf
import pickle
import codecs


def batch_generator(arr, n_seqs, n_steps):
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    while True:
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

def create_batches(arr, batch_size, seq_length):
    arr = copy.copy(arr)
    num_batches = int(len(arr) / (batch_size * seq_length))

    xdata = arr[:num_batches * batch_size * seq_length]
    ydata = np.copy(xdata)
    # ydata为xdata的左循环移位，例如x为[1,2,3,4,5]，y就为[2,3,4,5,1]
    # 因为y是x的下一个字符
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    # x_batches 的 shape 就是 223 × 50 × 100
    x_batches = np.split(xdata.reshape(batch_size, -1),
                              num_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1),
                              num_batches, 1)
    return x_batches, y_batches


def next_batch(x_batches, y_batches, pointer):
    x, y = x_batches[pointer], y_batches[pointer]
    pointer += 1
    return x, y, pointer




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
        if word in self.char2int:
            return self.char2int[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int2char[index]
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
