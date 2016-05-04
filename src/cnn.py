__author__ = 'insookyo'

import tensorflow as tf
import numpy as np
import os
from utility import load_images

class DataLoader(object):
    def __init__(self):
        self.train_dir = os.path.join(os.path.dirname(__file__), '../data', 'train')
        self.xs, self.ys = load_images(self.train_dir, mode="train")
        self.xs, self.ys = self.shuffle_data(self.xs, self.ys)
        self.data_idx = 0

    def generate_batch(self, batch_size):
        try:
            self.xs[self.data_idx+batch_size]
        except IndexError as e:
            self.xs, self.ys = self.shuffle_data(self.xs, self.ys)
            self.data_idx = 0

        batch_x = self.xs[self.data_idx:self.data_idx+batch_size]
        batch_y = self.ys[self.data_idx:self.data_idx+batch_size]
        self.data_idx+=batch_size

        return batch_x, batch_y

    def shuffle_data(self, xs, ys):
        random_idx = np.random.choice(range(len(xs)), len(xs), replace=False)
        return xs[random_idx, :], ys[random_idx, :]

# Create model
class CNN(object):
    def __init__(self, sess, learning_rate, training_iters, batch_size, display_step, n_input, n_classes, dropout):
        self.sess = sess
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        self.n_input = n_input
        self.n_classes = n_classes
        self.dropout = dropout


    def conv2d(self, img, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))


    def max_pool(self, img, k):
        return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv_net(self):
        # Reshape input picture
        self.x_4d = tf.reshape(self.x, shape=[-1, 64, 64, 1])
        # Convolution Layer
        self.conv1 = self.conv2d(self.x_4d, self._weights['wc1'], self._biases['bc1'])
        # Max Pooling (down-sampling)
        self.conv1 = self.max_pool(self.conv1, k=2)
        # Apply Dropout
        self.conv1 = tf.nn.dropout(self.conv1, self.keep_prob)

        # Convolution Layer
        self.conv2 = self.conv2d(self.conv1, self._weights['wc2'], self._biases['bc2'])
        # Max Pooling (down-sampling)
        self.conv2 = self.max_pool(self.conv2, k=2)
        # Apply Dropout
        self.conv2 = tf.nn.dropout(self.conv2, self.keep_prob)

        # Fully connected layer
        self.dense1 = tf.reshape(self.conv2, [-1, self._weights['wd1'].get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
        self.dense1 = tf.nn.relu(tf.add(tf.matmul(self.dense1, self._weights['wd1']), self._biases['bd1'])) # Relu activation
        self.dense1 = tf.nn.dropout(self.dense1, self.keep_prob) # Apply Dropout

        # Output, class prediction
        out = tf.add(tf.matmul(self.dense1, self._weights['out']), self._biases['out'])
        return out

    def build_graph(self):
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        # Store layers weight & bias
        self._weights = {
            'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32])), # 5x5 conv, 1 input, 32 outputs
            'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])), # 5x5 conv, 32 inputs, 64 outputs
            'wd1': tf.Variable(tf.random_normal([16*16*64, 1024])), # fully connected, 7*7*64 inputs, 1024 outputs
            'out': tf.Variable(tf.random_normal([1024, self.n_classes])) # 1024 inputs, 10 outputs (class prediction)
        }

        self._biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
        # Construct model
        self.pred = self.conv_net()

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))

    def train(self):
        print "graph is building..."
        self.build_graph()
        print "graph is built"
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        # Define Saver
        self.saver = tf.train.Saver()

        tf.initialize_all_variables().run()
        print "data is loading..."
        data_loader = DataLoader()
        print "data is loaded"
        step = 1
        # Keep training until reach max iterations
        while step * self.batch_size < self.training_iters:
            batch_xs, batch_ys =  data_loader.generate_batch(self.batch_size)

            # Fit training using batch data
            self.sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: self.dropout})
            if step % self.display_step == 0:
                print "step : %d (%f%%)" %(step, float(step * self.batch_size)/self.training_iters,)
                # Calculate batch accuracy and loss
                acc, loss = self.sess.run([self.accuracy, self.cost], feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: 1.})
                print "Iter " + str(step*self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
            step += 1

            if step % 10 == 0:
                self.save(step//10)

        print "Optimization Finished!"
        # Calculate accuracy for 256 mnist test images
        print "Testing Accuracy:", self.sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: 1.})

    def inference(self, x):
        y = np.zeros(shape=[x.shape[0], self.n_classes])

        self.build_graph()
        # Define Saver
        self.saver = tf.train.Saver()

        if self.load() is True:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        return self.sess.run(self.dense1, feed_dict={self.x: x, self.y: y, self.keep_prob: 1.})

    def save(self, step):
        """ """
        model_name ='cnn.model'
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'assets')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self):
        """ """
        print(" [*] Reading checkpoints...")

        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'assets')

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

if __name__ == '__main__':
    import time
    # Parameters
    learning_rate = 0.01
    training_iters = 100000
    batch_size = 128
    display_step = 10

    # Network Parameters
    n_input = 64*64 # MNIST data input (img shape: 28*28)
    n_classes = 62 # MNIST total classes (0-9 digits)
    dropout = 0.5 # Dropout, probability to keep units

    is_train = True

    with tf.Session() as sess:
        cnn = CNN(sess, learning_rate, training_iters, batch_size, display_step, n_input, n_classes, dropout)

        if is_train is True:
            start_time = time.time()
            cnn.train()
            print("--- %s seconds ---" % (time.time() - start_time))
            print("--- %s min ---" % ((time.time() - start_time)/float(60)))
        # else:
        #     # input_ids = np.asarray([0, 1, 2, 3, 4])
        #     input_ids = None
        #     embeddings = item2vec.inference(input_ids=input_ids)
        #     # plot_with_labels(embeddings, sampling="assigned", method="only_color")
        #     get_similar_items(embeddings)

