# VCOM 2019/2020

Curricular Unity: VCOM - Computer Vision<br>
Lective Year: 2019/2020

## Project 1: Detection of road signs

The goal of this project was to detect road signs in images and classify them according to their color and shape, in the following classes:<br>
- Red circle<br>
- Blue circle<br>
- Blue Square / Rectangle<br>
- Red Triangle<br>

We developed a system using **Python** and OpenCV**, to detect all these shapes, with the following steps:
1. Smoothing of the image using Gaussian function.
2. Color segmentation of the image.
3. Thresholding of the image.
4. Detection of Edges with the Canny Edge Detector algorithm.
5. Detection of the image contours and respective analysis.

For the detection of circles, the approach was slightly different, since in the 5th step, we used the contours to draw the external contours found
in the image on a blank image and used the Hough Circles algorithm to detect circles in this image, which gave us better results than simply using 
this algorithm on the original image and allowed us to always identify the outer circle on road signs that have a white circle in the center and a red circle as a border.
It also reduced the number of false circles identified in the image.

The system identifies multiple road signs, STOP signs, some poorly illuminated signs, partially occluded signs when it comes to circles and also slanted signs.

In this image, there is an example of the whole process for the detection of a triangular road sign:

![](https://github.com/SmilingOwl/VCOM-19_20/blob/master/img/triangle%20detection%20process.JPG)

In this image we have an example of the detection of 2 red circles that are badly illuminated and partially occluded:

![](https://github.com/SmilingOwl/VCOM-19_20/blob/master/img/circle%20detection.JPG)

For a more detailed overview of the project, please check the respective report [here](https://github.com/SmilingOwl/VCOM-19_20/blob/master/project1/docs/VCOM-report.pdf).

## Project 2: Classification and Segmentation of Skin Lesions

This project had 3 tasks:
- Comparison between different architectures for binary classification of images.
- Analysis of results of the best approached developed in the previous task for multi class classification.
- Segmentation of skin lesions in images.

### Task 1

For this first task, we developed 2 approaches:
- Bag of Visual Words using K-means for the vocabulary creation, SIFT for feature extraction and SVM for classification. In this approach, we studied different values for the size of 
the vocabulary and different kernels for the SVM. We arrived at the conclusion that the best results were achieved with 300 visual words and with the RBF kernel.
- VGG-16 with transfer learning. We used data augmentation and dropout to avoid overfitting; class weights and bias initialization to battle class imbalance.

Between these approaches, we achieved the best results using the CNN approach, with VGG-16, which was to be expected, since CNNs are the current state-of-art for image classification.

### Task 2

For this task, we used the VGG-16 architecture which resulted in the best results in the first task.

### Task 3

For this task, we used a UNET architecture to achieve segmentation, using Batch Normalization in the contraction path to increase the stability of the network, and with different
 loss functions, that we compared and arrived at the conclusion that Dice loss worked the best for our data sets.

For a more detailed overview of this project, check the respective article [here](https://github.com/SmilingOwl/VCOM-19_20/blob/master/project2/docs/Report.pdf).
