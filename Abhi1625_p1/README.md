# Project 1: MyAutoPano! for the course CMSC-733
Homography between 2 image frames is defined as the projective transformation between these frames and is one of the very key concepts in Computer Vision. In this project we used Homography to warp multiple images and stitch a panorama using three different techniques - Traditional approach using feature matching and RANSAC, Supervised approach to predict a 4 point parametrization of Homography between two images and an Unsupervised approach to predict Homography without the presence of a ground truth. Some of the output panoramas generated with the traditional approach are shown below:
<img src="Draft/mypano.png" align="center" alt="Pano1" height="300"/>


##Running Phase 1 - Traditional Approach:
- Run the file Wrapper.py 
The file should run smoothly to generate the results for the Train Set 2. 
**Note**: In case the file doesn't run check that the relative path `../Data/Train/Set2` exists. 

- To run the Wrapper.py for a different data set, use command line argument `BasePath`
```
python Wrapper.py --BasePath= Path/To/Files
```

<!-- ##Running Phase 2:
-In the file `Train.py` change the line
```
from Network.*** import CIFAR10Model
```
to 
```
from Network.ResNext import CIFAR10Model
```
inorder to run the ResNext Model. Change this lines for all the models you want to run: 
Network1, ResNet, ResNext, DenseNet. Do the same for `Test.py` file. -->
