from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

embedsz = 30
batchsz = 20
hsize = 100
learningrate = 1e-4
num_epochs = 1
train = open('train.txt', 'r')
vocab = {}
vocabsz = 0
trainingwords = []
testingwords = []
for line in train:
    for word in line.split():
        if vocab.has_key(word) == False:
            vocab[word] = vocabsz
            vocabsz += 1
        trainingwords.append(vocab[word])

test = open('test.txt', 'r')
for line in test:
    for word in line.split():
        if vocab.has_key(word) == False:
            vocab[word] = vocabsz
            vocabsz += 1
        trainingwords.append(vocab[word])

train.close()
test.close()

def get_batch(x):
    input = trainingwords[:x]
    output = trainingwords[1:x+1]
    return input, output

sess = tf.InteractiveSession()
print(vocabsz)

input = tf.placeholder(tf.int32, [None])
output = tf.placeholder(tf.int32, [None])
E = tf.Variable(tf.random_uniform([vocabsz, embedsz], -1, 1))
embd = tf.nn.embedding_lookup(E, input)

W1 = tf.Variable(tf.truncated_normal([embedsz, hsize], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([1, hsize], stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([hsize, vocabsz], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([1, hsize], stddev=0.1))
h1 = tf.nn.relu(tf.matmul(embd, W1) + b1)
h2 = tf.matmul(h1, W2) + b2 # the logits
ERROR = tf.nn.sparse_softmax_cross_entropy_with_logits(h2, output)
train_step = tf.train.AdamOptimizer(learningrate).minimize(ERROR)
sess.run(tf.initialize_all_variables())
for e in range(num_epochs):
    TOT_ERROR = 0
    x = batchsz
    while x < len(trainingwords):
        inpt, outpt = get_batch(x)
        dontcare, err = sess.run([train_step, ERROR], feed_dict = {input: inpt, output: outpt})
        err = np.average(err)
        TOT_ERROR += err
        perplexity = np.exp([TOT_ERROR / x])
        if x % 100:
            print(perplexity, x)
        x += batchsz

for e in range(num_epochs):
    TOT_ERROR = 0
    x = batchsz
    while x < len(testingwords):
        inpt, outpt = get_batch(x)
        dontcare, err = sess.run([ERROR], feed_dict = {input: inpt, output: outpt})
        err = np.average(err)
        TOT_ERROR += err
        perplexity = np.exp([TOT_ERROR / x])
        if x % 100:
            print(perplexity, x)
        x += batchsz
    print(perplexity)