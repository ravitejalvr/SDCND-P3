### Project Description

In this project, Convolutional Neural Network is trained such that steering angles were predicted real time.

The following steps were followed:

1. Udacity Car driving simulator is driven in testing mode for collecton of good driving behavior data.
2. Then, a Convolution Neural Network model is built with Nvidia style of layers and implementation
3. The model is trained with good driving data for measuring the weights which are required for the model
4. Then, the saved model is then used in real-time testing conditions for testing effectiveness of the model  

The following files were submitted as a part of this project
* `README.md` Summarizing the results and explaining framework
* `model.py` script which creates model.h5 file. required for generating the video
* `model.h5` File which contains weights and details of trained Convolutional Neural Network
* `drive.py` For connecting to simulator and driving the car in autonomous mode
* `video.mp4` Recorded video of the run

[img_normal](images/normal.jpg?raw=true "Before Augmented")
[img_augmented](images/augmented.jpg?raw=true "Augmented")

##### Output video
[![IMAGE ALT TEXT](http://img.youtube.com/vi/39qgJ9l7mj4/0.jpg)](https://youtu.be/39qgJ9l7mj4)

#### Collection of training data
Training data was chosen to keep the vehicle driving on the road. I collected the training data in the following manner:
- two laps of center lane driving
- two laps of center lane driving in reverse manner
- one lap of recovery driving from the sides
- one lap focusing on driving smoothly around curves

#### Data Preprocessing
- The images are cropped and resized to 66x200 so that its same as NVIDIA model
- Then, images were normalized for zero mean and unit variance (divided by 255 and subtracted 1.0) for faster convergence and no     overfitting.

#### Overfitting of Model
Overfitting can be huge issue while using Neural Networks and hence to curb this issue the following methodologies were implemented:

- The image is normalized to zero mean and unit variance. 
- Dropout of probability 0.5 is implemented for having randomly dropped some of the nets.

#### Image Augmentation
The following techniques were used for data augmentation 

- Randomly increase steering anglr For left image, right image by 0.25 towards center of lane.
- Randomly flip images such that model is trained both sides.
- Change brightness randomly such that image is adjucted to all conditions.

The images are also randomly shuffled and 20 & data is used in validation set.

Here is an example of images before and after image augmentation:
Before Augmentation:
![img_normal]

After Augmentation
![img_augmented]

#### Convolutional Neural Network Features

My CNN is based on this paper: [NVIDIA's steering angle prediction model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

I trained this network to minimize the MSE (Mean-Square Error) of estimated steering angle. The figure below shows the network architecture: 

It can be seen that the architcture contains 9 layers, out of which 5 layers are 2D Convolutional Layers, one flatten Layer, followed by 4 fully connected layers. The input image is processed before inputting to model by converting it to YUV image.

The layers are summarized in the table below:

|Layer (type)           |          Output Shape     |    Params # |  
|:---|:---:|---:|
| Normalization (Lambda)|      (None, 66, 200, 3)   |   0         |
| conv2d_1 (Conv2D)     |      (None, 31, 98, 24)   |   1824      |
| conv2d_2 (Conv2D)     |      (None, 14, 47, 36)   |   21636     |
| conv2d_3 (Conv2D)     |      (None, 5, 22, 48)    |   43248     |
| conv2d_4 (Conv2D)     |      (None, 3, 20, 64)    |   27712     |
| conv2d_5 (Conv2D)     |      (None, 1, 18, 64)    |   36928     |
| flatten_1 (Flatten)   |      (None, 1152)         |   0         |
| dense_1 (Dense)       |      (None, 100)          |   115300    |
| dense_2 (Dense)       |      (None, 50)           |   5050      |
| dense_3 (Dense)       |      (None, 10)           |   510       |
| dense_4 (Dense)       |      (None, 1)            |   11        |       

The model used an ADAM optimizer which is given inputs as mean squared error for minimizing.

The tuned model is then saved into model.h5 file, which is then used in drive.py file for driving autonomously.

#### Results
The model can drive the course smoothly without going off the track.  
The video file attached in this file explains it. 
Also Youtube video link is given below:
[![IMAGE ALT TEXT](http://img.youtube.com/vi/39qgJ9l7mj4/0.jpg)](https://youtu.be/39qgJ9l7mj4)
