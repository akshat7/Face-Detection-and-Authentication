import os
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

def getImagesWithID(path, str):
	imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
	print imagePaths
	faces = []
	IDs = []
	for imagePath in imagePaths:
		faceImg = Image.open(imagePath).convert('L')
		faceNp = np.array(faceImg, 'uint8')
		faceNp = cv2.resize(faceNp, (250, 250))
		ID = int(os.path.split(imagePath)[-1].split('.')[1])
		faces.append(faceNp)
		IDs.append(ID)
		cv2.imshow(str, faceNp)
		cv2.waitKey(10)

	return np.array(IDs), faces

def training():
	recognizer_pca = cv2.face.createEigenFaceRecognizer()
	recognizer_lda = cv2.face.createFisherFaceRecognizer()
	path = 'Dataset'
	IDs, faces = getImagesWithID(path, 'Training')
	cv2.destroyAllWindows()
	recognizer_pca.train(faces, IDs)
	eigenvectors = recognizer_pca.getEigenVectors();
	mean = recognizer_pca.getMean()
	mean = np.reshape(mean, (-1, 250))
	mean = np.uint8(mean)
	cv2.imshow('Mean', mean)
	cv2.waitKey(10000)
	print eigenvectors[0].shape, mean.shape
	recognizer_pca.save('recognizer/TrainingData1.yml')
	cv2.destroyAllWindows()

def testing():
	path = "TestingDataset"
	IDs, faces = getImagesWithID(path, 'Testing')

training()
