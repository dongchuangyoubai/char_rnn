# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

class Model:
    def __init__(self, num_classes, log_dir, batch_size=50, seq_length=64,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5,
                 use_embedding=False, embedding_size=128):
        if sampling is True:
            seq_length, batch_size = 1, 1
        else:
            seq_length, batch_size = seq_length, batch_size

        self.num_classes = num_classes
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size
        self.log_dir = log_dir

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()
        self.summaries = tf.summary.merge_all()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.seq_length, self.batch_size), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(
                self.seq_length, self.batch_size), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # 对于中文，需要使用embedding层
            # 英文字母没有必要用embedding层
            if not self.use_embedding:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_lstm(self):
        # 创建单个cell并堆叠多层
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state = cell.zero_state(self.seq_length, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

            # 通过lstm_outputs得到概率
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')
            tf.summary.histogram('logits', self.logits)

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)
            tf.summary.histogram('loss', self.loss)

    def build_optimizer(self):
        # 使用clipping gradients
        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        # train_op = tf.train.AdamOptimizer(self.learning_rate)
        # self.optimizer = train_op.apply_gradients(zip(grads, tvars))
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.minimize(self.loss)

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n, log_dir):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())


            writer = tf.summary.FileWriter(log_dir)
            writer.add_graph(sess.graph)

            # Train network
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                summ, batch_loss, new_state, _ = sess.run([
                                                    self.summaries,
                                                    self.loss,
                                                    self.final_state,
                                                    self.optimizer],
                                                    feed_dict=feed)
                # print(summ)
                writer.add_summary(summ, step)
                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
            writer.close()

    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))  # for prime=[]
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, vocab_size)
        # 添加字符到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))