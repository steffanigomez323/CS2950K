"""
bigram_lm.py

Implementation of Feed-Forward Language Model, with Embeddings.
"""
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_size', 30, 'Word Embedding size.')
tf.app.flags.DEFINE_integer('hidden_size', 100, 'Size of the hidden layer.')
tf.app.flags.DEFINE_integer('batch_size', 20, 'Batch Size for training.')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Learning Rate for Adam Optimizer.')
tf.app.flags.DEFINE_integer('eval_every', 100, 'Print Loss every eval_every batches.')

TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"


class BigramLM():
    def __init__(self, embedding_sz, hidden_sz, vocab_sz, learning_rate):
        """
        Instantiate a BigramLM Model, with the necessary hyperparameters.

        :param embedding_sz: Size of the word embeddings.
        :param hidden_sz: Size of the hidden ReLU layer.
        :param vocab_sz: Size of the vocabulary.
        """
        self.embedding_sz, self.hidden_sz, self.vocab_sz = embedding_sz, hidden_sz, vocab_sz
        self.learning_rate = learning_rate

        # Setup Placeholders
        self.X = tf.placeholder(tf.int32, [None])                  # Shape [bsz]
        self.Y = tf.placeholder(tf.int32, [None])                  # Shape [bsz]

        # Instantiate Network Parameters
        self.instantiate_weights()

        # Build the Logits
        self.logits = self.inference()

        # Build the Loss Computation
        self.loss_val = self.loss()

        # Build the Training Operation
        self.train_op = self.train()

    def instantiate_weights(self):
        """
        Instantiate the network Variables, for the Embedding, Hidden, and Output Layers.
        """
        # Embedding
        self.E = self.weight_variable([self.vocab_sz, self.embedding_sz], 'E')

        # Hidden Layer
        self.relu_W = self.weight_variable([self.embedding_sz, self.hidden_sz], 'ReLU_W')
        self.relu_B = self.bias_variable([self.hidden_sz], 'ReLU_B')

        # Output Layer
        self.output_W = self.weight_variable([self.hidden_sz, self.vocab_sz], 'Output_W')
        self.output_B = self.bias_variable([self.vocab_sz], 'Output_B')

    def inference(self):
        """
        Build the inference computation graph for the model, going from the input to the output
        logits (before final softmax activation).
        """
        emb = tf.nn.embedding_lookup(self.E, self.X)                    # Shape [bsz, embedding_sz]
        relu = tf.nn.relu(tf.matmul(emb, self.relu_W) + self.relu_B)    # Shape [bsz, hidden_sz]
        return tf.matmul(relu, self.output_W) + self.output_B           # Shape [bsz, vocab_sz]

    def loss(self):
        """
        Build the cross-entropy loss computation graph.
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.Y))

    def train(self):
        """
        Build the training operation, using the cross-entropy loss and an Adam Optimizer.
        """
        opt = tf.train.AdamOptimizer(self.learning_rate)
        return opt.minimize(self.loss_val)

    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

def read(train_file, test_file):
    """
    Read and parse the file, building the vectorized representations of the input and output.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train_x, train_y, test_x, test_y, vocab_size
    """
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []
    with open(train_file, 'r') as f:
        for line in f:
            tokens = line.split()
            train_data.extend(tokens)
            for tok in tokens:
                if tok not in vocabulary:
                    vocabulary[tok] = vocab_size
                    vocab_size += 1
    with open(test_file, 'r') as f:
        for line in f:
            tokens = line.split()
            test_data.extend(tokens)

    # Sanity Check, make sure there are no new words in the test data.
    #assert reduce(lambda x, y: x and (y in vocabulary), test_data)

    # Vectorize, and return output tuple.
    train_data = map(lambda x: vocabulary[x], train_data)
    test_data = map(lambda x: vocabulary[x], test_data)
    return np.array(train_data[:-1]), np.array(train_data[1:]), np.array(test_data[:-1]), \
        np.array(test_data[1:]), vocab_size


# Main Training Block
if __name__ == "__main__":
    # Preprocess the Data, generate the X, Y for both Train, Test
    train_x, train_y, test_x, test_y, voc_sz = read(TRAIN_FILE, TEST_FILE)
    num_train, bsz = len(train_x), FLAGS.batch_size

    # Launch Tensorflow Session
    print 'Launching Session!'
    with tf.Session() as sess:
        # Instantiate Model
        bigram = BigramLM(FLAGS.embedding_size, FLAGS.hidden_size, voc_sz, FLAGS.learning_rate)

        # Initialize all Variables
        sess.run(tf.initialize_all_variables())

        print 'Starting Training!'
        loss, counter = 0.0, 0
        for start, end in zip(range(0, num_train, bsz), range(bsz, num_train + bsz, bsz)):
            curr_loss, _ = sess.run([bigram.loss_val, bigram.train_op],
                                    feed_dict={bigram.X: train_x[start:end],
                                               bigram.Y: train_y[start:end]})
            loss, counter = loss + curr_loss, counter + 1
            if counter % FLAGS.eval_every == 0:
                print 'Batch {} Train Perplexity:'.format(counter), np.exp(loss / counter)

        test_loss = sess.run(bigram.loss_val, feed_dict={bigram.X: test_x, bigram.Y: test_y})
        print 'Test Perplexity:', np.exp(test_loss)