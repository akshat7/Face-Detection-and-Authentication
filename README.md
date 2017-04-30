# Face Detection and Authentication

## Methodology

### Problem Definition
Face Detection, Verification and Authentication: The subject stands near the camera, and a picture is captured. The faces in that image are detected and verified to be known or unknown. If known, finally the subject is authenticated and allowed to enter from the door.

### Dataset creation
There are total 10 classes, corresponding to 10 different people, consisting solely of their face images. Out of these 10 classes, 5 are trained with “Known” tag, 3 with “Unknown” tag (NOTA: None Of The Above), and 2 are test classes for unknown. Each class consists of the same person giving different facial expressions (including side poses as well) and in different light conditions, namely-
* Natural Low Light
* Natural High Light
* Artificial Low Light
* Artificial High Light

### Deciding the parameters and number of the training input image dataset
The size of each image is equal, having dimensions of [40 X 40], which are reshaped and feeded into the Principle Component Analysis Model in the form of a [1600 X 1] vector each. There are in total 10 classes, consisting of 20 images each - 5 different expressions (including different angles as well) for each of the 4 different light conditions. So, in total, there are 8 X 10 = 80 training images.

## Preprocessing 

### FACE DETECTION
Read the input image data and detect the face boundaries in each image using Frontal-Face Haar Cascades.
### Resize
The image is cropped around the detected boundaries, and is then resized to [40X40] dimension. 
### Grayscale
The image is then converted from RGB to Gray.
### Histogram Equalization
The final grayscale face then undergo histogram equalization.
### Reshape the images
Images having dimension [40 X 40] are reshaped into [1600 X 1] for feeding them into PCA.
Save the reshaped image data and their corresponding class labels.
### Training the image dataset on the combined model
The images are trained on a combined PCA-LDA-ANN model, in which PCA has been used for dimensionality reduction, LDA for Discriminant Analysis and feature extraction, and ANN for classification.
### PCA
Feed the final histogram equalized and reshaped data into the PCA model. The Principal Components taken are 80 out of 1600. The reason for taking these many components is that the sum of the variance ratio was coming to be 0.99485.
### LDA
Feed the PCA Eigenfaces into LDA. The output of LDA is fed into the ANN.
### ANN
Our ANN consists of 3 layers, the first layer consists of the output of the LDA, the hidden layer consists of 20 nodes (experimentally determined), and the output layer consists of scores corresponding to each class.
### Capturing (real-time) the test image
Camera switched on, image gets captured, Camera switches off. The captured image is saved.
### Detecting faces from the captured image
From the captured image, detect the face boundaries of each face in the image using Frontal-Face Haar Cascades, and the detected faces are saved.
### Preprocessing the detected face images
The preprocessing of the detected test face images is done in a similar way, as described before for the training images.
### Testing
Feeding the images into the trained model for testing.
### Classification
Classifying the output as “Known” or “Unknown”.
