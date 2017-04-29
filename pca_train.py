import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.externals import joblib


n_components = 80

final_data = np.load('final_data.npy')
y = np.load('final_labels.npy')
# print final_data.shape

pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(final_data)

n_features = final_data.shape[1]
eigenfaces = pca.components_.reshape((n_components, 40, 40))
X_train_pca = pca.transform(final_data)
# print X_train_pca.shape
# print sum(pca.explained_variance_ratio_), pca.components_.shape

s = joblib.dump(pca, 'pca_trained_clf.pkl')
Xs = joblib.dump(X_train_pca, 'X_train_pca.pkl')