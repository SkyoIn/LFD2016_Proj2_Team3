from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn import svm
import os
from utility import load_images
from PIL import Image
import numpy as np

train_dir = os.path.join(os.path.dirname(__file__), '../data', 'train')
test_dir = os.path.join(os.path.dirname(__file__), '../data', 'test')


def load_data():
    train_X, train_Y = load_images(train_dir, mode="train")
    test_X, test_Y = load_images(test_dir, mode="test")
    # train_X, train_Y = test_X, test_Y

    return train_X, train_Y, test_X, test_Y

def feature_engineering(train_X):
    return train_X


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_data()
    print "data load complete"
    train_X = feature_engineering(train_X)

    print 'training phase'
    clf = svm.LinearSVC().fit(train_X, train_Y)
    print 'test phase'
    predicts = clf.predict(test_X)

    # measure function
    print 'measure phase'
    print confusion_matrix(test_Y, predicts)
    print f1_score(test_Y, predicts, average=None)
    print precision_score(test_Y, predicts, average=None)
    print recall_score(test_Y, predicts, average=None)
    print accuracy_score(test_Y, predicts)
