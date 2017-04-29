import cv2
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib


X_train_lda = joblib.load('X_train_lda.pkl')
y = np.load('final_labels.npy')

print X_train_lda.shape

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
        		hidden_layer_sizes=(20), random_state=1).fit(X_train_lda, y)

s = joblib.dump(mlp, 'mlp_trained_clf.pkl')

# print mlp.classes_, mlp.predict_proba(X_test_lda)
# print mlp.n_layers_, mlp.n_outputs_
# print mlp.score(X_test_lda, np.array([3]))