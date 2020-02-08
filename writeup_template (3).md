# **Traffic Sign Recognition** 

## Writeup

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


[image4]: ./examples/ex.png "Traffic Signs"


---

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)



### Data Set Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES
Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the pandas shape method might be useful for calculating some of the summary results.


* The training set contains 34799 examples
* The validation set contains 4410 examples 
* The test set contains 12630 examples
* The shape of each image in the train, validation and test set is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the 43 classes.

 Train distribution = (180 1980 2010 1260 1770 1650  360 1290 1260 1320 1800 1170 1890 1920  690
  540  360  990 1080  180  300  270  330  450  240 1350  540  210  480  240
  390  690  210  599  360 1080  330  180 1860  270  300  210  210)
  
  
 Test distribution = (60 720 750 450 660 630 150 450 450 480 660 420 690 720 270 210 150 360
 390  60  90  90 120 150  90 480 180  60 150  90 150 270  60 210 120 390
 120  60 690  90  90  60  90)


![Train dist]: [./examples/train.PNG] 
"Train data distribution"
![Valid dist]: [./examples/valid.PNG] 
"Validation data distribution"
![Test dist]: [./examples/test.PNG] 
"Test data distribution"



### Design and Test a Model Architecture

#### Preprocessing the dataset

Initially I tried out different permutations of the following preprocessing techniques:

1) Conversion to grayscale
2) Normalizing with the formula abs((img - 128)/128)
3) Normalizing with the formula abs((img - mean(img))/stddev(img))

But for some reason they didnt give me any better results.

So I just reshuffled the training dataset.
It also supports my objective of building an end-to-end deep learning pipeline utilizing the colors of the traffic signs.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


|Layer                  |  Description                                  |
|:---------------------:|:----------------------------------------------|
|Input              	| 32x32x3 RGB Image 	                        |
|Convolution 5 x 5,     |                                               |
|  6 output channels;   |                                               |
|Activation - ReLu 		| 1x1 strides, valid padding, outputs 28x28x6	|
|Convolution 5x5,       |                                               |
|12 output channels;    |                                               |
|Activation - ReLu	 	| 2x2 strides,valid padding, outputs 13x13x12	|
|Convolution 5x5,       |                                               |
|16 output channels;    |                                               |
|Activation - ReLu	 	| 1x1 strides, valid padding, outputs 9x9x16	|
|Max Pooling 2x2    	| 2x2 strides , valid padding, outputs 4x4x16	|
|Flatten            	| Outputs 1x256                              	|
|Dropout 1          	| Keep probability – 0.8                     	|
|Fully Connected layer 1|                                               |
|Activation - ReLu  	| 256x120                                   	|
|Fully Connected layer 2|                                               |
|Activation - ReLu   	| 120x100                                   	|
|Dropout 2          	| Keep probability – 0.8                    	|
|Fully Connected layer3	|                                               |
|Activation - ReLu    	| 100x84                                    	|
|Fully Connected layer 	| 84x43                                     	|
|Activation - Softmax	| 84x43                                     	|





#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The objective was to create an end to end deep learning architecture. The classical LeNet-5 architecture was working fine but was giving an accuracy of around 87% on the validation set. So the idea to add an extra convolutional and fully connected layer was executed. The rest of the architectural aspects were included by hit and trial experimentation with the objective of achieving a 94-95 % accuracy rate on the validation set.

The batch size was chosen as 128. Adam Optimizer was used as per the default recommendations.

The learning rate was manually tuned and was set as 0.0006 based on observed learning patterns.

The network architecture being a bit complex showed overfitting so drop outs were pushed in at places. Even though the overfitting was not rectified completely but some improvement was definitely seen.

The epoch was set as 85 to fulfill the objective of achieving 94-95% accuracy more securely.

Even though there's plenty of scope for improvement in the model architecture for getting better results in this case, but as of now this model fulfills the baseline objectives.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9927 
* validation set accuracy of 0.9396  
* test set accuracy of 0.9286 


* What was the first architecture that was tried and why was it chosen?

The first architecture chosen was the recommended LeNet 5 atchitecture as it was recommended and seemed to work fine in the classroom videos.

* What were some problems with the initial architecture?

There was no problems as such but I wanted to experiment with a complex network of my own which is more likely to fulfill the purpose of finding an end to end solution - i.e without the involvement of any preprocessing.


The objective was to create an end to end deep learning architecture. The classical LeNet-5 architecture was working fine but was giving an accuracy of around 87% on the validation set. So the idea to add an extra convolutional and fully connected layer was executed. The rest of the architectural aspects were included by hit and trial experimentation with the objective of achieving a 94-95 % accuracy rate on the validation set.

The batch size was chosen as 128. Adam Optimizer was used as per the default recommendations.

The learning rate was manually tuned and was set as 0.0006 based on observed learning patterns.

The network architecture being a bit complex showed overfitting so drop outs were pushed in at places. Even though the overfitting was not rectified completely but some improvement was definitely seen.

The epoch was set as 85 to fulfill the objective of achieving 94-95% accuracy more securely.

Even though there's plenty of scope for improvement in the model architecture for getting better results in this case, but as of now this model fulfills the baseline objectives.

 

### Test a Model on New Images


Here are five German traffic signs that I found on the web:

![img_1][my_test_images/00002.jpg] ![img 3][my_test_images/00005.jpg] ![img 5][my_test_images/00009.jpg] 
![img 2][my_test_images/00011.jpg] ![img 4][my_test_images/00012.jpg]

The first image might be difficult to classify because ...



Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No passing   			| No passing 									|
| Ahead only			| Ahead only									|
| 60 km/h	      		| 60 km/h   					 				|
| Priority Road			| Priority Road      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

For all the images 


Softmax matrices: (  4.06350726e-37   5.98420675e-37   9.04541066e-33   6.76123647e-29
    1.02519494e-31   6.02810529e-37   0.00000000e+00   9.65672573e-37
    5.42796547e-34   1.00000000e+00   1.34064087e-23   0.00000000e+00
    6.81983397e-28   3.78539821e-28   6.93057363e-32   1.59002252e-24
    9.73427378e-24   1.51547166e-28   0.00000000e+00   0.00000000e+00
    0.00000000e+00   0.00000000e+00   0.00000000e+00   2.77315285e-38
    0.00000000e+00   1.02123705e-31   0.00000000e+00   0.00000000e+00
    1.43703158e-37   0.00000000e+00   0.00000000e+00   0.00000000e+00
    5.61278946e-38   9.43791131e-32   4.56648331e-34   4.76158415e-24
    7.57229308e-34   0.00000000e+00   3.32079410e-28   5.72190117e-37
    1.90615750e-35   5.10455659e-32   2.50582886e-33]
    
    
 ( 1.60326244e-33   4.54214907e-24   2.62125136e-26   0.00000000e+00
    1.25570921e-35   2.63559994e-33   0.00000000e+00   1.34297426e-24
    1.95278162e-29   5.54448328e-29   1.39522383e-35   2.10544462e-36
    1.00000000e+00   3.91635613e-23   9.36275947e-30   4.40938483e-28
    5.78204493e-29   1.69798983e-31   0.00000000e+00   0.00000000e+00
    0.00000000e+00   0.00000000e+00   0.00000000e+00   1.42923238e-34
    0.00000000e+00   1.14959257e-31   1.12859924e-34   0.00000000e+00
    1.94231916e-31   6.52505779e-35   4.08333789e-28   1.53954001e-31
    1.13204733e-29   4.98062787e-31   0.00000000e+00   2.34999118e-31
    0.00000000e+00   1.47980457e-31   1.61832213e-34   8.70284060e-32
    2.31493164e-19   1.44246078e-33   7.27635662e-38]
    
    
 ( 3.75071099e-37   5.29137925e-18   2.16855131e-15   1.00000000e+00
    1.32586061e-24   7.43579330e-12   2.27421441e-33   2.23846350e-23
    1.04703200e-30   4.45593666e-20   1.53498790e-18   4.22930819e-32
    1.05572389e-21   1.46654589e-26   2.20465443e-12   5.61610342e-25
    3.47757017e-38   4.74272141e-28   1.29693422e-28   1.70462091e-33
    1.54154792e-26   0.00000000e+00   1.95603454e-34   2.00863246e-27
    0.00000000e+00   1.63811435e-22   1.99969041e-24   0.00000000e+00
    8.20493723e-23   1.25914942e-26   1.72998815e-38   3.62087950e-29
    2.31252860e-21   3.29124483e-21   9.47706745e-22   1.90093272e-24
    5.65375881e-19   4.08336086e-32   1.17564382e-22   5.55360369e-30
    7.08916851e-31   0.00000000e+00   4.71361795e-25]
    
    
 ( 2.80560868e-29   1.16578991e-33   0.00000000e+00   0.00000000e+00
    1.01134031e-37   0.00000000e+00   0.00000000e+00   0.00000000e+00
    0.00000000e+00   1.81828325e-30   2.23823638e-35   0.00000000e+00
    8.30995739e-30   1.31489343e-27   1.18742579e-29   6.55814814e-27
    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
    0.00000000e+00   3.36746547e-31   2.82864033e-35   1.00000000e+00
    3.37133623e-30   0.00000000e+00   3.31426812e-33   0.00000000e+00
    0.00000000e+00   0.00000000e+00   5.56834977e-29]
    
    
 ( 2.39302883e-20   2.10318945e-13   4.12169856e-19   7.30163407e-15
    1.13117735e-18   9.71164699e-17   4.01721177e-35   2.66810586e-23
    7.90081746e-24   6.60922641e-21   1.43849528e-23   7.15184686e-27
    1.91771320e-17   1.31169752e-19   1.00000000e+00   2.91020730e-18
    4.51931094e-27   1.02299991e-15   1.69888713e-20   0.00000000e+00
    1.12402399e-18   1.43313969e-29   1.08473194e-23   1.49497315e-28
    3.99360752e-34   1.12728778e-25   3.12794996e-21   1.85700577e-37
    9.48449103e-20   5.28770166e-20   1.47754983e-29   4.18421973e-30
    2.18356825e-22   7.24132912e-23   1.31443046e-22   4.23544366e-17
    4.33329110e-12   1.02198266e-21   4.30890239e-20   2.78574488e-26
    2.79473827e-19   6.68919198e-31   8.86943926e-26]

top 5 values:

  
  
  No passing:       (  1.00000000e+00,   7.57358168e-22,   1.45269026e-22, 1.60891248e-24,   1.31774130e-26) | (9, 16, 15, 35, 10)
          
          
  Priority road:    (  1.00000000e+00,   1.24846168e-23,   1.56838760e-27,  1.43046340e-29,   1.00734005e-29) | (12, 40,  1, 13,  9)
          
          
  60 Km/h:          (  1.00000000e+00,   4.69886308e-09,   9.66738516e-13, 5.03058492e-14,   2.20322722e-14)  | ( 3,  2,  1,  5, 14)
          
 
  Ahead only:       (  1.00000000e+00,   1.00187320e-18,   2.07574047e-19, 9.77429441e-20,   3.40394451e-20)  | (35, 36, 15,  9,  0)
          
  
  Stop:             (  1.00000000e+00,   9.96980234e-22,   5.25872697e-23, 2.47291722e-24,   2.19866973e-24)  | (14,  3,  5, 36, 38)
          
       

By looking at the softmax and probability values it seems the model is able to determine with pretty firm conviction.

I guess these pictures look a lot like the ones we have in our dataset. If we have some pictures which are newer or having different color shades or being a bit more 'generic' in terms of pixel patterns, the model may not be able to predict correctly as there is apparently an overfitting.



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


