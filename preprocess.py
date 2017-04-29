import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

path = 'Dataset'
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
print imagePaths
faces = []
IDs = []
data = []
for imagePath in imagePaths:
    if imagePath != 'Dataset/.DS_Store':
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        faceNp = cv2.resize(faceNp, (40, 40))
        dataNp = faceNp.reshape((1600,))
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        data.append(dataNp)
final_images = np.asarray(faces)
final_data = np.asarray(data)
n_features = final_data.shape[1]
y = np.asarray(IDs)
n_classes = 3
np.save('final_data', final_data)
np.save('final_labels', y)