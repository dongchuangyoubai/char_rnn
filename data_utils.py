import numpy as np
import copy
import tensorflow as tf
import pickle
import codecs


def batch_generator(arr, batch_size, seq_length):
    arr = copy.copy(arr)
    # print(arr[:10])
    # print(len(arr))
    n_batches = int(len(arr) / (batch_size * seq_length))
    # print(n_batches)
    arr = arr[:batch_size * seq_length * n_batches]
    arr = arr.reshape((seq_length, -1))
    # print(arr.shape)
    while True:
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], batch_size):
            x = arr[:, n: n + batch_size]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y


class TextConverter(object):
    def __init__(self, text=None, filename=None):
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
    tf.flags.DEFINE_string('name', 'default', 'name of the model')
    tf.flags.DEFINE_integer('batch_size', 32, 'length of one seq')
    tf.flags.DEFINE_integer('seq_length', 100, 'number of seqs in one batch')
    tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
    tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
    tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
    tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
    tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
    tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
    tf.flags.DEFINE_string('input_file', 'data/shakespeare.txt', 'utf8 encoded text file')
    tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
    tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
    tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
    tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()

    converter = TextConverter(text)
    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.batch_size, FLAGS.seq_length)

