# Akshat Aggarwal
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
def createDataset():
    path = 'Dataset_raw'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print imagePaths
    images = []
    final_images = []

    for imagePath in imagePaths:
        if imagePath != 'Dataset_raw/.DS_Store':
            images.append([os.path.join(imagePath, f) for f in os.listdir(imagePath)])
    #     print imagePath
    print len(images)
    for i in range(len(images)):
        count = 0
        for item in images[i]:
            print item.split('/')[-1]
            if item.split('/')[-1] != '.DS_Store':
    #         faceImg = Image.open(item).convert('L')
                faceImg = cv2.imread(item)
        #         print type(faceImg)
                faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)
                faces = faceDetect.detectMultiScale(gray, 1.3, 5)
                sampleNumber = 0
                for x, y, w, h in faces:
                    # print count
                    count += 1
                    sampleNumber += 1
                    name = "Dataset/"+str(i+1)+"/"+str(count)+".png"
                    final_images.append(name)
                    cv2.imwrite(name, gray[y:y+h, x:x+w])
    #         faceNp = np.array(faceImg, 'uint8')
    #         faceNp = cv2.resize(faceNp, (40, 40))
    #         dataNp = faceNp.reshape((1600,))
    

    ################################

def saveDataset():
    faces = []
    IDs = []
    data = []

    path = 'Dataset'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print imagePaths
    images = []
    final_images = []

    for imagePath in imagePaths:
        if imagePath != 'Dataset/.DS_Store':
            images.append([os.path.join(imagePath, f) for f in os.listdir(imagePath)])

    for i in range(len(images)):
        for item in images[i]:
            if item.split('/')[-1] != '.DS_Store':
                print item

                # img = Image.open(item).convert('L')
                img = cv2.imread(item,0)
                hist,bins = np.histogram(img.flatten(),256,[0,256])
                cdf = hist.cumsum()
                cdf_normalized = cdf *hist.max()/ cdf.max()
                cdf_m = np.ma.masked_equal(cdf,0)
                cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
                cdf = np.ma.filled(cdf_m,0).astype('uint8')
                img2 = cdf[img]

                faceImg = img2
                faceNp = np.array(faceImg, 'uint8')
                faceNp = cv2.resize(faceNp, (40, 40))
                dataNp = faceNp.reshape((1600,))
        #         ID = int(os.path.split(item)[-1].split('.')[1])
                ID = i+1
                faces.append(faceNp)
                IDs.append(ID)
                data.append(dataNp)
    final_images = np.asarray(faces)
    final_data = np.asarray(data)
    n_features = final_data.shape[1]
    y = np.asarray(IDs)
    np.save('final_data', final_data)
    np.save('final_labels', y)

# createDataset()
saveDataset()