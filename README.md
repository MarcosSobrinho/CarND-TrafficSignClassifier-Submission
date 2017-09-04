## **Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[training_data]: ./writeup_images/training_data.png "Training Data"
[validation_data]: ./writeup_images/validation_data.png "Validation Data"
[testing_data]: ./writeup_images/testing_data.png "Testing Data"
[sign1]: ./web_images/einfache_vorfahrt.png "Einfache Vorfahrt"
[sign2]: ./web_images/einfahrt_verboten.png "Einfahrt verboten"
[sign3]: ./web_images/stopschild.png "Stop"
[sign4]: ./web_images/vorfahrt_gewaehren.png "Vorfahrt gewaehren"
[sign5]: ./web_images/vorfahrtstrasse.png "Vorfahrtstrasse"

## Rubric Points

---
### Files Submitted

This repository contains the Ipython Notebook as well as an HTML with the implementation. 

### Dataset Exploration

#### Dataset Summary

I used the pandas and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory Visualization

The following bar charts will show the distributions of classes in the datasets:

* Training data
![alt text][training_data]

* Validation data 
![alt text][validation_data]

* Testing data
![alt text][testing_data]

The distributions look quite similar. It can be seen, that only a few images of the class 0 are in the datasets compared to the classes 1 and 2. Furthermore the peaks are all more or less in the same positions. The classes 1, 2, 10, 38  are  some obvious examples.

### Design and Test a Model Architecture

#### Preprocessing

The two steps I took into consideration for the image preprocessing are:

* Normalization with 0 mean and values between -1 and 1
* Grayscaling

Normalization is a general approach for data preprocessing. The normalized data performed a little bit better (about 1% better validation accuracy) than the unnormalized data.

Considering grayscaling, I actually didn't want to use it, because it made sense to me to also provide the network with information about color distribution. After all many signs can already be roughly classified by only looking at the colors and not necessarily at the shape. However, LeNet performed slightly better with grayscaled images. Possibly because convolutional layer have an easier time training neurons that detect geometrical features. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I didn't care about the order of the two mentioned steps for preprocessing, because they are - mathematically speaking - linear operations. So the result of the normalization steps shouldn't depend on the order of these operations. However, first grayscaling and then normalizing is probably more efficient, due to the amount of computations.

#### Model Architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, vald padding, outputs 28x28x15 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x15 				|
| Convolution 5x5     	| 1x1 stride, vald padding, outputs 10x10x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs5x5x20 				|
| Fully connected		| outputs 120, dropout with 0.5 while training						|
| RELU					|												|
| Fully connected		| outputs 84, dropout with 0.5 while training								|
| RELU					|												|
| Fully connected		| outputs 43								|


#### Model Training & Solution Approach

To train the model, I used a modified version of LeNet. While the [Udacity classroom implementation](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) performs outstandingly with with the MNIST-Dataset, the current model was too weak to reach an validation accuracy of at least 0.93. It only reached 0.89.

The only hyper parameter I changed is the number of Epochs. I doubled it to 20, just to see if there is anything to be gained with more than 10 epochs. However the Learning doesn't improve significantly after the tenth epoch.

MNIST only consideres 10 different classes, whereas here there are 43 classes. This is why the output layer has to be adjusted. 

My first thought was to increase the number of neurons in the convolutional layer. The reason for this is the intuition that neurons in convolutional layers correspond to image filters for specific features. This means that an increased number of neurons in the convolutional layer is able to filter more features. 

However, an increased number of neurons leads to higher risk of an overfitting model. This is why I added dropout to the last two hidden layers with a dropout probability of50%.

These two measures already lead to a validation accuracy of 96.1%. However, I wanted to add L2-regularization to the networks parameters. At first my weighting factor for the regularization loss was too high with 0.001, and the accuracy was decreasing. I went down to 0.0001, and my validation reached 96.8% validation accuracy. Without L2-regularization the network even reached a little bit more than 97% validation accuracy, however I insisted in keeping L2-regularization at least a little bit, because it leads to smoother classification boundaries.

I also experimented with Batch Normalization. Sadly, it made things worse in this case so I took it out again. 

My final model results were:

* validation set accuracy of 96.8%
* test set accuracy of 94.8%

###Test a Model on New Images

#### Acquiring New Images

Here are five German traffic signs I randomly cut out of pictures found in the internet. When cutting out the traffic signs, I tried to get the format described in the [German traffic signs page](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Imageformat).

![alt text][sign1] 

![alt text][sign2] 

![alt text][sign3] 

![alt text][sign4] 

![alt text][sign5]

Even though the images have never been seen by the network, they should be easily recognized and classified correctly. All of the images have the same basically the same format as in the training database. Furthermore the signs are very sharp and can be seen clearly. Also the illumination conditions are very good. The only "noise" existing is some reflection in the second sign.

#### Performance on New Images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   									| 
| No entry    			| No entry 										|
| Stop				| Stop											|
| Yield	      		| Yield				 				|
| Priority road			| Priority road     							|


The model was able to correctly guess 5 of the 5 traffic signs, so it has a 100% accuracy for this randomly chosen little testing set. This comares quite well to the original test set prediction, since the network scored a 94.8% accuracy. With these clear images at least 4 out of 5 images were expected to be classified correctly.

#### Model Certainty - Softmax Probabilities

For all images the network is absolutely sure that the prediction is correct. As can be seen in the notebook, the network predicts the images with an accuracy of 100% and the other remaining 4 of the top 5 results with basically 0%. Either the network is very powerfull or the images are very close to training images. However, this would still be a big coincidence, because I chose the images arbitrarily. 

Here is the example of the first sign, however the softmax probabilities are the same for every other example

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Right-of-way at the next intersection  									| 
| .00     				| Beware of ice/snow 										|
| .00					| Pedestrians											|
| .00	      			| General Caution					 				|
| .00				    | Double curve      							|

