# Detecting vehicles using machine learning and computer vision

The final project from [Udacity self-driving car course](http://udacity.com/drive) is creating a software pipeline which is capable of identifying cars in a video from a front-facing camera on a car.

![alt tag](https://github.com/bdjukic/CarND-Vehicle-Detection/raw/master/readme_images/1.jpg)
A snapshot from the final output of the project

The course material is suggesting the usage of somewhat outdated approach for detecting vehicles which I figured out in the middle of the project by reading this [great paper](https://t.co/VFxlrhQ70C) on the state-of-the-art computer vision for autonomous vehicles. Small snippet from that paper:

> With the work of Dalal & Triggs (2005), linear Support<br>
Vector Machines (SVMs), that maximizes the margin of all<br>
samples from a linear decision boundary, in combination with<br>
Histogram of Orientation (HOG) features have become popular<br>
tools for classification. However, all previous methods rely on<br>
hand-crafted features that are difficult to design. **With the renaissance<br>of deep learning, convolutional neural networks have<br>automated this task while significantly boosting performance**.

As it turns out, Deep Neural Networks are outperforming the approach which I have used (Linear Support Vector Machines in combination with Histogram of Oriented Gradients). I will defiantly go back to this project and try out some of the top performers in this list on the same problem:
Table taken from [https://arxiv.org/pdf/1704.05519.pdf](https://arxiv.org/pdf/1704.05519.pdf)

![alt tag](https://github.com/bdjukic/CarND-Vehicle-Detection/raw/master/readme_images/2.jpg)
The process of detecting the vehicles on the road could be summed up in the following steps:

![alt tag](https://github.com/bdjukic/CarND-Vehicle-Detection/raw/master/readme_images/3.jpg)
### Training data analysis

Training data is provided by Udacity and it consists of images of [cars](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) from different angles (8792) and [non-car](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) (8968) objects. Here are two samples:
![alt tag](https://github.com/bdjukic/CarND-Vehicle-Detection/raw/master/readme_images/4.jpg)
Examples from the training data set

### Feature extraction

In order to detect a car on the image, we need to identify **feature(s)** which uniquely represent a car. We could try using simple template matching or relaying on color features but these methods are not robust enough when it comes to changing perspectives and shapes of the object.

In order to have a robust feature set and increase our accuracy rate we will be using [Histogram of Oriented Gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) (HOG). This feature descriptor is much more resilient to the dynamics of the traffic. In essence, you should **think of features as footprints** of the objects you are interested in.

[Scikit-image](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog) python library provides us with the necessary API for calculating HOG feature. I have used [YCrCb](https://en.wikipedia.org/wiki/YCbCr) color space and all its channels as inputs for HOG features extraction. I have tried other color spaces, but YCrCb gave me the best accuracy when training my prediction model. Here's a sample of vehicle and non-vehicle image with HOG features from the same images as above:
![alt tag](https://github.com/bdjukic/CarND-Vehicle-Detection/raw/master/readme_images/5.jpg)
Extracted HOG features from sample training data

HOG feature extraction was based on 9 orientations, 8 pixels per cell and 2 cells per block. Increasing orientations and pixel per cell parameters did improve prediction time but the accuracy rate of the model went down.

### Model training

In order to detect the car based on our feature set, we would need a prediction model. For this particular case we will be using [Linear Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) (Linear SVMs). It is a supervised learning model which will be able to classify whether something is a car or not after we train it.

HOG features have been scaled to zero mean and unit variance using [Standard Scaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

I have divided provided data set into training set (80%) and testing set (20%). Images have been shuffled as well before kicking off the training session. In the end Linear SVMs model with the extracted HOG features on YCrCb color space reached **98.06%** accuracy rate.

### Sliding windows

Once we have the prediction model, it's time to use it on our test images. Prediction model will be applied in a special technique called Sliding Windows. With this technique we will be running prediction model on sub-regions of the images which is divided into a grid.

In order to increase the robustness of this approach we will be adding multiple grids which will be traversed by the prediction model. We are doing this since cars can appear on the image in various sizes depending on its location on the road.

![alt tag](https://github.com/bdjukic/CarND-Vehicle-Detection/raw/master/readme_images/6.jpg)
Multi-window approach for sliding window technique

This is how this concept looks like when it is applied on our test image:
![alt tag](https://github.com/bdjukic/CarND-Vehicle-Detection/raw/master/readme_images/7.jpg)
I have used different window sizes (from 128x128 for area closer to the car and 64x64 for area further away from the vehicle). Windows overlap is set to 65%.

### Eliminating false positives

In order to improve the accuracy of the final output we will be trying to find multiple hits for the object of interest in the similar area. This approach is equivalent to creating a heat map.

The next step is introducing threshold which needs to be met in order for a specific hit count from the heat map to be accepted as a detected car. In our case threshold has a value of 2.

This is an example of applying the heat map and thresholding it to a specific value.
![alt tag](https://github.com/bdjukic/CarND-Vehicle-Detection/raw/master/readme_images/8.jpg)
### Final outcome

The final outcome is not perfect but the pipeline itself is showing good potential.
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/aP4ZFcPH7wM/0.jpg)](https://www.youtube.com/watch?v=aP4ZFcPH7wM)
### Conclusion

If I would get a chance to re-do this project, I would probably go for a Deep Neural Network approach. I have spent considerable amount of time searching for the right color space, HOG parameters and windows slide's size.

The performance of the pipeline is not great and can be improved. Deep Neural Network approach would have better performance numbers.

Since this car detection approach is based on camera it's prone to usual challenges with this kind of sensor (bad visibility, reflections, etc.).

Project as usual can be found on my [Github](https://github.com/bdjukic/CarND-Vehicle-Detection) profile.
