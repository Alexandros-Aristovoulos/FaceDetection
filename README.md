# FaceDetection

### Description
The program tracks faces in a picture or in a video and draws a rectangle around them. It is also able to identify the direction the face is looking and draws 
a different color rectangle around the forward, left and right looking face.

### How it works
Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, 
"Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade function is trained from 
a lot of positive and negative images. It is then used to detect objects in other images.

Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. 
Then we need to extract features from it. For each feature, the classifier finds the best threshold which will classify the faces to positive and 
negative. Obviously, there will be errors or misclassifications but we select the features with minimum error rate, which means they are the features that 
most accurately classify the face and non-face images.

However, we can't just use the final classifier to check each area of the picture for all the features since that procedure would be extremely time consuming. For 
this reason since most of the image is a non-face region it is a better idea to have a simple method to check if a window is not a face region. If it is not, 
discard it in a single shot, and we don't process it again. Instead, we focus on regions where there can be a face. This way, we spend more time checking possible face regions.

For this reason exists the concept of Cascade of Classifiers which instead of applying all features on a window, the features are grouped into different stages 
of classifiers and applied one by one. (Normally the first few stages will contain very many fewer features). If a window fails the first stage, we discard it and 
we don't consider the remaining features on it. If it passes, we apply the second stage of features and continue the process. The window which passes all stages is a face region.

### Example
![Input Image](/test.jpg)
![Output Image](/testOutput.png)

The program draws a green rectnagle for front faces, blue rectnagle for left faces and a red one for right faces

### Dependencies
- [Numpy](https://numpy.org/install/)
- [OpenCV](https://pypi.org/project/opencv-python/)
