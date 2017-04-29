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


# pca = joblib.load('pca_trained_clf.pkl')
# lda = joblib.load('lda_trained_clf.pkl')
# mlp = joblib.load('mlp_trained_clf.pkl')

# faceImg = Image.open('test11.png').convert('L')

# try:
#     faceImg = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)
# except:
#     pass

# faceNp = np.array(faceImg, 'uint8')
# faceNp = cv2.resize(faceNp, (40, 40))

# X_test = faceNp.reshape((1,1600))
# X_test_pca = pca.transform(X_test)
# X_test_lda = lda.transform(X_test_pca)

# print "LDA prediction: ", lda.predict(X_test_pca)
# print "PCA prediction: ", mlp.predict(X_test_lda)



def get_image():
    return_value, im = camera.read()
    return im


camera_port = 0
ramp_frames = 30 # Number of frames to throw away while the camera adjusts to light levels
camera = cv2.VideoCapture(camera_port)

for i in xrange(ramp_frames):
    temp = get_image()
    
camera_capture = get_image()
print type(camera_capture)
cv2.imwrite("img.png", camera_capture)
del(camera)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(camera_capture, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, 1.3, 5)
print faces
sampleNumber = 0
for x, y, w, h in faces:
        sampleNumber += 1 
        cv2.imwrite("Detected/User."+str(sampleNumber) + ".png", gray[y:y+h, x:x+w])

pca = joblib.load('pca_trained_clf.pkl')
lda = joblib.load('lda_trained_clf.pkl')
mlp = joblib.load('mlp_trained_clf.pkl')

path = 'Detected'
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

faces = []
IDs = []
data = []
print imagePaths
for imagePath in imagePaths:
    if imagePath != 'Detected/.DS_Store':
        # faceImg = Image.open(imagePath).convert('L')
        faceImg = cv2.imread(imagePath,0)
        try:
            faceImg = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)
        except:
            pass

        img = faceImg
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf *hist.max()/ cdf.max()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        img2 = cdf[img]

        faceNp = np.array(img2, 'uint8')
        faceNp = cv2.resize(faceNp, (40, 40))
        dataNp = faceNp.reshape((1600,))
#         ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
#         IDs.append(ID)
        data.append(dataNp)
final_images = np.asarray(faces)
final_data = np.asarray(data)

# faceImg = Image.open('111.png').convert('L')



# faceNp = np.array(faceImg, 'uint8')
# faceNp = cv2.resize(faceNp, (40, 40))

# X_test = faceNp.reshape((1,1600))
# X_test_pca = pca.transform(X_test)
X_test_pca = pca.transform(final_data)
X_test_lda = lda.transform(X_test_pca)
# print lda
# print X_test_lda
# print "LDA prediction: ", lda.predict(X_test_pca)
# print "MLP prediction: ", mlp.predict(X_test_lda)
final_result = lda.predict(X_test_pca)
count = 0
for item in final_result:
    print item
    if item == 1 or item == 2 or item == 3 or item == 4 or item == 5:
        count += 1

if count == 0:
    print "Unknown"
else:
    print "Known"


