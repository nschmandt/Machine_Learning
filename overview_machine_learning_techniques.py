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

#read in the data

trainData,tuneData,testData=read_mnist("/home/nick/Class_Unsupervised_learning_data_analysis/data/original_distribute.mat")
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

#first classification: PCA anlaysis, see chart for selection of optimal n_components

pca=decomposition.PCA(n_components=30)
pca.fit(train_x_sample[::50])
pca_tune=pca.transform(tune_x_sample)
pca_test=pca.transform(test_x_sample)
pca_train=pca.transform(train_x_sample)

#best error prediction was around .1 for clean, .2 for noisy

best_error_tune, pred = linear_svm(pca_train, train_y_sample, pca_tune, tune_y_sample, pca_tune, 0)

#Isomapping the data, see plot for identifying idea number of clusters

clf = manifold.Isomap(30, n_components=15)
clf.fit(train_x_sample[::50])
train_iso = clf.transform(train_x_sample)
tune_iso = clf.transform(tune_x_sample)
test_iso = clf.transform(test_x_sample)

best_error_tune, pred = linear_svm(train_iso, train_y_sample, tune_iso, tune_y_sample, test_iso, 0)

#t-SNE implementation is below, and it does not seem to be highly successful, best error is .5 on the train set

tsne=manifold.TSNE(perplexity=30, n_components=2, init="pca", n_iter=3000)
z_tsne_train=tsne.fit_transform( np.asfarray(train_x_sample[::50], dtype="float") )
z_tsne_tune=tsne.fit_transform( np.asfarray(tune_x_sample[::10], dtype="float") )
z_tsne_test=tsne.fit_transform( np.asfarray(test_x_sample[::10], dtype="float") )

best_error_tune, pred = linear_svm(z_tsne_train, train_y_sample[::50], z_tsne_tune, tune_y_sample[::10], z_tsne_test, 0)

# A very crude MDS implementation is below, that simply calculates the difference between each average image and tries to get the answer.
#accuracy is about .5 just straight up, but higher if SVM is attempted. Note there's no learning here, so train is as good as test.

# average_images=average_digit_images(train_x_image, train_y_sample)
#
predicted_train_value=np.zeros((train_x_image.shape[0],1))
predicted_tune_value=np.zeros((tune_x_image.shape[0],1))
predicted_test_value=np.zeros((test_x_image.shape[0],1))

for i in range(1, train_x_image.shape[0]):
    dif_matrix = np.zeros((10))
    for j in range(0, 10):
        dif_matrix[j]=(sum(sum(np.absolute(average_images[j]-train_x_image[i]))))
    if len((np.where(dif_matrix == min(dif_matrix))[0]))==1:
        diff_value=min(dif_matrix)
        predicted_train_value[i]=(np.where(dif_matrix==min(dif_matrix))[0])
# If you got two even likelihood answers, just go with the lowest
    else:
        temp=np.where(dif_matrix==min(dif_matrix))[0]
        predicted_train_value[i]=temp[1]

for i in range(1, tune_x_image.shape[0]):
    dif_matrix = np.zeros((10))
    for j in range(0, 10):
        dif_matrix[j]=(sum(sum(np.absolute(average_images[j]-tune_x_image[i]))))
    if len((np.where(dif_matrix == min(dif_matrix))[0]))==1:
        diff_value=min(dif_matrix)
        predicted_tune_value[i]=(np.where(dif_matrix==min(dif_matrix))[0])
#If you got two even likelihood answers, just go with the lowest
    else:
        temp=np.where(dif_matrix==min(dif_matrix))[0]
        predicted_tune_value[i]=temp[1]

print(total_train_error=np.sum(predicted_train_value!=train_y_sample)/predicted_train_value.shape[0])
best_error_tune, pred = linear_svm(predicted_train_value, train_y_sample, predicted_tune_value, tune_y_sample, predicted_test_value, 0)
