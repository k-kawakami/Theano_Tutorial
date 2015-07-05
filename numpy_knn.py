import time

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score

mnist = fetch_mldata('MNIST original')
X, y = mnist.data.astype("float32"), mnist.target.astype("int32")
X /= X.sum(axis=1)[:, numpy.newaxis]
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

#Numpy
ts = time.time()
closest_sample = numpy.argmax(numpy.dot(test_x, train_x.T), axis=1) 
numpy_time = time.time() - ts

pred = train_y[closest_sample]
numpy_f1 = f1_score(test_y, pred, average='macro')

print "Numpy:: %.3f [sec], F1: %.3f"%(numpy_time, numpy_f1)
