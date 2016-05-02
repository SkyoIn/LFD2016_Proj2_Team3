from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn import svm

from utility import load_images


train_dir = '../data/train'
test_dir = '../data/test'

train_X, train_Y = load_images(train_dir)
test_X, test_Y = load_images(test_dir)

# training phase
## TODO
# change training method
print 'training phase'
clf = svm.SVC()
clf.fit(train_X, train_Y)

# predict phase
print 'test phase'
predicts = clf.predict(test_X)

# measure function
print 'measure phase'
print confusion_matrix(test_Y, predicts)
print f1_score(test_Y, predicts, average=None)
print precision_score(test_Y, predicts, average=None)
print recall_score(test_Y, predicts, average=None)
print accuracy_score(test_Y, predicts)
