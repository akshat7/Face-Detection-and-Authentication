import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
from sklearn.externals import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


X_train_pca = joblib.load('X_train_pca.pkl')
y = np.load('final_labels.npy')

print X_train_pca.shape

lda = LinearDiscriminantAnalysis( priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001).fit(X_train_pca, y)

X_train_lda = lda.transform(X_train_pca)
# print X_train_lda.shape

s = joblib.dump(lda, 'lda_trained_clf.pkl')
Xs = joblib.dump(X_train_lda, 'X_train_lda.pkl')
