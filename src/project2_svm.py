from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn import svm
import os
from utility import load_images
from cnn import CNN
import tensorflow as tf


train_dir = os.path.join(os.path.dirname(__file__), '../data', 'train')
test_dir = os.path.join(os.path.dirname(__file__), '../data', 'test')


def load_data(resize_shape):
    train_X, train_Y = load_images(train_dir, mode="train", one_hot=False, resize_shape=resize_shape)
    test_X, test_Y = load_images(test_dir, mode="test", one_hot=False,resize_shape=resize_shape)
    # train_X, train_Y = test_X, test_Y

    return train_X, train_Y, test_X, test_Y

def main():
    resize_shape = 64
    print "data is loading..."
    train_X, train_Y, test_X, test_Y = load_data(resize_shape)
    print "data is loaded"
    print "feature engineering..."
    learning_rate = 0.01
    training_iters = 100000
    batch_size = 128
    display_step = 10

    # Network Parameters
    n_input = resize_shape*resize_shape # MNIST data input (img shape: 28*28)
    n_classes = 62 # MNIST total classes (0-9 digits)
    dropout = 0.5 # Dropout, probability to keep units

    with tf.Session() as sess:
        cnn = CNN(sess, learning_rate, training_iters, batch_size, display_step, n_input, n_classes, dropout,resize_shape)
        train_X = cnn.inference(train_X)
        test_X = cnn.inference(test_X)

    print "feature engineering is complete"

    print 'training phase'
    clf = svm.SVC().fit(train_X, train_Y)
    print 'test phase'
    predicts = clf.predict(test_X)

    # measure function
    print 'measure phase'
    print confusion_matrix(test_Y, predicts)
    print f1_score(test_Y, predicts, average=None)
    print precision_score(test_Y, predicts, average=None)
    print recall_score(test_Y, predicts, average=None)
    print accuracy_score(test_Y, predicts)

if __name__ == '__main__':
    main()
