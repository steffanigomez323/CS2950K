from __future__ import division
import numpy as np
import tensorflow as tf
import re
from collections import Counter
import operator

embeddingsz = 50
batchsz = 100
num_steps = 20
keepprob = 0.5
learning_rate = 1e4
dropout = 0.5
hiddensz = 256
epochs = 5

def basic_tokenizer(sentence, word_split=re.compile(b"([.,!?\"':;)(])")):
    """
    Very basic tokenizer: split the sentence into a list of tokens, lowercase.
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(word_split, space_separated_fragment))
    return [w.lower() for w in words if w]

def readfile(train_file):
    asterisks = 0
    startreading = False
    words = []
    with open(train_file, 'r') as training:
        for line in training:
            tokens = basic_tokenizer(line)
            if startreading == False and asterisks < 2:
                for token in tokens:
                    if token == "***":
                        asterisks += 1
                        if asterisks == 2:
                            startreading = True
            elif startreading:
                for i in range(len(tokens)):
                    if tokens[i] == "***":
                        # stop reading
                        asterisks += 1
                        startreading = False
                        break
                    if tokens[i] == ".":
                        tokens.insert(i + 1, "STOP")
                if asterisks > 2:
                    break
                words.extend(tokens)
        c = Counter(words)
        sorted_c = sorted(c.items(), key=operator.itemgetter(1), reverse=True)[:8000]
        vocabulary = dict(zip([str(i[0]) for i in sorted_c], [integer for integer in range(8000)]))
        vocabulary["*UNK*"] = 8000
        for index in range(len(words)):
            if vocabulary.has_key(words[index]) == False:
                words[index] = "*UNK*"
        words_list = map(lambda x: vocabulary[x], words)
        training_data, testing_data = words_list[:int(len(words_list) * .9)], words_list[int(len(words_list)*.9):]
        return np.array(training_data[:-1]), np.array(training_data[1:]), np.array(testing_data[:-1]), np.array(testing_data[:-1]), 8001


# class LSTM():
#     def __init__(self, vocabsz):
#
#         self.X = tf.placeholder(tf.int32, [batchsz, num_steps])
#         self.Y = tf.placeholder(tf.int32, [batchsz, num_steps])  # Shape [batchsz, num_steps]
#         self.keep_prob = tf.placeholder(tf.float32)
#
#         self.E = tf.Variable(tf.truncated_normal([vocabsz, embeddingsz], stddev=0.1))
#         self.W = tf.Variable(tf.truncated_normal([hiddensz, vocabsz], stddev=0.1))
#         self.b = tf.Variable(tf.constant(0.1, shape=[vocabsz]))
#
#         self.lstm = tf.nn.rnn_cell.BasicLSTMCell(hiddensz)
#         self.initial_state = self.lstm.zero_state(batchsz, tf.float32)
#
#         embd = tf.nn.embedding_lookup(self.E, self.X)  # Shape [vocabsz, embeddingsz, batchsz]
#         embd_drop = tf.nn.dropout(embd, self.keep_prob)
#
#         cell_output, out_state = tf.nn.dynamic_rnn(self.lstm, embd_drop, initial_state=self.initial_state)
#
#         self.final_state = out_state
#
#         output = tf.reshape(cell_output, [batchsz*num_steps, hiddensz])
#         self.logits = tf.matmul(output, self.W) + self.b
#
#         # Build the Loss Computation
#         self.loss_val = self.loss()
#
#         # Build the Training Operation
#         self.train_op = self.train()
#
#
#     def loss(self):
#         reshaped_y = tf.reshape(self.Y, [batchsz*num_steps])
#         loss1 = tf.nn.seq2seq.sequence_loss_by_example(
#             [self.logits],
#             [reshaped_y],
#             [tf.ones([batchsz*num_steps], tf.float32)])
#         return tf.reduce_sum(loss1) / batchsz
#
#     def train(self):
#         opt = tf.train.AdamOptimizer(learning_rate)
#         return opt.minimize(self.loss_val)

if __name__ == "__main__":
    train_x, train_y, test_x, test_y, vocabsz= readfile('dracula.txt')
    num_train = len(train_x)

    # Launch Tensorflow Session
    print 'Launching Session!'
    with tf.Session() as sess:
        # Instantiate Model
        #lstm = LSTM(vocabsz)

        X = tf.placeholder(tf.int32, [batchsz, num_steps])
        Y = tf.placeholder(tf.int32, [batchsz, num_steps])  # Shape [batchsz, num_steps]
        keep_prob = tf.placeholder(tf.float32)

        E = tf.Variable(tf.truncated_normal([vocabsz, embeddingsz], stddev=0.1))
        W = tf.Variable(tf.truncated_normal([hiddensz, vocabsz], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[vocabsz]))

        lstm = tf.nn.rnn_cell.BasicLSTMCell(hiddensz, state_is_tuple=True)
        initial_state = lstm.zero_state(batchsz, tf.float32)

        embd = tf.nn.embedding_lookup(E, X)  # Shape [vocabsz, embeddingsz, batchsz]
        embd_drop = tf.nn.dropout(embd, keep_prob)

        cell_output, out_state = tf.nn.dynamic_rnn(lstm, embd_drop, initial_state=initial_state)

        #final_state = out_state

        output = tf.reshape(cell_output, [batchsz * num_steps, hiddensz])
        logits = tf.matmul(output, W) + b

        # Build the Loss Computation
        reshaped_y = tf.reshape(Y, [batchsz * num_steps])
        loss1 = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [reshaped_y],
            [tf.ones([batchsz * num_steps], tf.float32)])
        loss = tf.reduce_sum(loss1) / batchsz

        # Build the Training Operation
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    #
    # def loss(self):
    #     reshaped_y = tf.reshape(self.Y, [batchsz * num_steps])
    #     loss1 = tf.nn.seq2seq.sequence_loss_by_example(
    #         [self.logits],
    #         [reshaped_y],
    #         [tf.ones([batchsz * num_steps], tf.float32)])
    #     return tf.reduce_sum(loss1) / batchsz
    #
    #
    # def train(self):
    #     opt = tf.train.AdamOptimizer(learning_rate)
    #     return opt.minimize(self.loss_val)


        # Initialize all Variables
        sess.run(tf.initialize_all_variables())

        print 'Starting Training!'
        loss_val, counter = 0.0, 0
        num_batch = int((num_train - num_steps)) / batchsz
        curr_state = [initial_state[0].eval(), initial_state[1].eval()]
        print num_batch
        #for start, end in zip(range(0, num_train, batchsz), range(batchsz, num_train + batchsz, batchsz)):
        for epoch in range(epochs):
            for i in range(int(num_batch)):
                x = []
                y = []
                #for i in range(num_steps):
                #    for j in range(start, end):
                for j in range(batchsz):
                    x.append(train_x[i * batchsz + j: i * batchsz + j + num_steps])
                    y.append(train_y[i * batchsz + j: i * batchsz + j + num_steps])
                        #x.append(train_x[j])
                        #y.append(train_y[j])
                x, y = np.array(x), np.array(y)
                #x.shape = (batchsz, num_steps)
                curr_loss, final_state, _ = sess.run([loss, out_state, train_op],
                                        feed_dict={X: x,
                                                   Y: y,
                                                   keep_prob: keepprob,
                                                   initial_state[0]: curr_state[0],
                                                   initial_state[1]: curr_state[1]})
                curr_state = [final_state[0], final_state[1]]
                loss_val, counter = loss_val + curr_loss, counter + 1
                iters = counter * num_steps
                if counter % 100 == 0:
                    print loss_val, iters
                    print 'Batch {} Train Perplexity:'.format(counter), np.exp(curr_loss / iters)

        test_loss, counter = 0.0, 0
        curr_state = [initial_state[0].eval(), initial_state[1].eval()]
        num_batch = int((len(test_x) - num_steps) / batchsz)
        print len(test_x)
        print num_batch
        for i in range(num_batch):
            x_test = []
            y_test = []
            for j in range(batchsz):
                x_test.append(test_x[i * batchsz + j: i * batchsz + j + num_steps])
                y_test.append(test_y[i * batchsz + j: i * batchsz + j + num_steps])
            x_test, y_test = np.array(x_test), np.array(y_test)
            loss_val, final_state = sess.run([loss, out_state], feed_dict={
                X: x_test,
                Y: y_test,
                keep_prob: 1,
                initial_state[0]: curr_state[0],
                initial_state[1]: curr_state[1]})
            curr_state = [final_state[0], final_state[1]]
            test_loss += loss_val
            counter += 1
            iters = counter * num_steps
            if counter % 100:
                print 'Batch {} Test Perplexity:'.format(counter), np.exp(loss_val / iters)

        #test_loss = sess.run(bigram.loss_val, feed_dict={bigram.X: test_x, bigram.Y: test_y})
        print 'Test Perplexity:', np.exp(test_loss / iters)



