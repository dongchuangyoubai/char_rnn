import tensorflow as tf
from data_utils import TextConverter, batch_generator
from model import Model
import os
import codecs

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('seq_length', 100, 'number of seqs in one batch')
tf.flags.DEFINE_integer('batch_size', 32, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', 'data/shakespeare.txt', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 14290, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1429, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 1429, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')


def main(_):
    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text)
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.batch_size, FLAGS.seq_length)
    print(converter.vocab_size())
    model = Model(converter.vocab_size(),
                    batch_size=FLAGS.batch_size,
                    seq_length=FLAGS.seq_length,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()
