# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

class Model:
    def __init__(self, num_classes, batch_size=64, seq_length=50,
                 rnn_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self.build_inputs()
        self.build_rnn()
        # self.build_loss()
        # self.build_optimizer()
        # self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope("inputs"):
            self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.seq_length], 'inputs')
            self.targets = tf.placeholder(tf.int32, [self.batch_size, self.seq_length], 'targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            if not self.use_embedding:
                # inputs shape [batch_size, seq_length, num_classes]
                self.inputs = tf.one_hot(self.inputs, self.num_classes)

    def build_rnn(self):
        def get_a_cell(rnn_size, keep_prob):
            cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        with tf.name_scope('rnn'):
            # multi rnn layers
            cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.rnn_size, self.keep_prob) for _ in range(self.num_layers)])
            # initial hidden state
            self.initial_state = cell.zero_state(self.batch_size, tf.float32)
            # get rnn outputs ang hidden state
            # outpus shape [batch_size, seq_length, num_classes]
            self.outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=self.initial_state)

            # 将多层rnn输出按axis=1 整合到一起 outpus shape [batch_size * num_classes, seq_length]
            self.outputs = tf.concat(self.outputs, 1)
            # outpus shape [, rnn_size]
            self.outputs = tf.reshape(self.outputs, [-1, self.rnn_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.rnn_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(self.outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits, name='predictions')







if __name__ == '__main__':
    m = Model(65)