# https://github.com/ArduPilot/MAVProxy/pull/258/commits/28231d4420f298cfff58865bb21e24155d9810b7

import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

rec = cv2.face.createLBPHFaceRecognizer(threshold=10000)
rec.load('recognizer/TrainingData.yml')

id = 0
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

while(True):
	ret, img = cam.read()
	# print "-------------", ret
	# print img
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# gray = cv2.resize(gray, (50, 50))
	faces = faceDetect.detectMultiScale(gray, 1.3, 5)

	for x, y, w, h in faces:
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		id, conf = rec.predict(gray[y:y+h, x:x+w])
		print id
		if id == 1:
			id = "Akshat"
		elif id == 2:
			id = "Arqum"
		elif id == 3:
			id = "Farzaaaaaaaan"
		elif id == -1:
			id = "Unknown" 
		# cv2.putText(img, str(id),(x, y+h), font,  (0, 255, 0))
		cv2.putText(img, str(id), (x, y+h), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (255, 0, 255))
	cv2.imshow("Face", img)
	if cv2.waitKey(1)==ord('q'):
		break 
cam.release()
cv2.destroyAllWindows()

