import sys
import numpy as np
import os

os.chdir("/home/nick/Class_Unsupervised_learning_data_analysis/data")

sys.path.append("/home/nick/Class_Unsupervised_learning_data_analysis/Python")
from mnist_twoview import read_mnist_twoview
from plot import *
from mnist import read_mnist
from utility import *
from linear_svm import linear_svm
from utility_sup import *
from sklearn.decomposition import KernelPCA

#read in the data

trainData,tuneData,testData=read_mnist("/home/nick/Class_Unsupervised_learning_data_analysis/data/noisy_distribute.mat")
test_x_sample = testData
test_x_image = np.reshape(test_x_sample, [test_x_sample.shape[0],28,28]) ### Reshape [#sample, 28*28] to [#sample, 28, 28]
test_x_image = np.swapaxes(test_x_image,1,2)
train_x_sample = trainData.images
train_x_image = np.reshape(train_x_sample, [train_x_sample.shape[0],28,28])
train_x_image = np.swapaxes(train_x_image,1,2)
train_y_sample = np.reshape(trainData.labels, [train_x_sample.shape[0]])
tune_x_sample = tuneData.images
tune_x_image = np.reshape(tune_x_sample, [tune_x_sample.shape[0],28,28])
tune_x_image = np.swapaxes(tune_x_image,1,2)
tune_y_sample = np.reshape(tuneData.labels, [tune_x_sample.shape[0]])



pca=decomposition.PCA(n_components=200)
pca.fit(train_x_sample[::50])
pca_tune=pca.transform(tune_x_sample)
pca_test=pca.transform(test_x_sample)
pca_train=pca.transform(train_x_sample)

clf = manifold.Isomap(300, n_components=250)
clf.fit(pca_train[::50])
train_iso = clf.transform(pca_train)
tune_iso = clf.transform(pca_tune)
test_iso = clf.transform(pca_test)


best_error_tune, pred = linear_svm(train_iso, train_y_sample, tune_iso, tune_y_sample, test_iso, 0)


f=open('kaggle_noisy.txt', 'w')
f.write('Target,Class\n')
for i in range(0, test_iso.shape[0]):

    f.write("Target %s,%s\n" %(i+1, pred[i]))
f.close()