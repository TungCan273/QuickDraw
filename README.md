# QuickDraw
# Description 
Quick, Draw! is an online game developed by Google that challenges players to draw a picture of an object or idea and then uses a neural network artificial intelligence to guess what the drawings represent. The AI learns from each drawing, increasing its ability to guess correctly in the future.The game is similar to Pictionary in that the player only has a limited time to draw (20 seconds).The concepts that it guesses can be simple, like 'foot', or more complicated, like 'animal migration'. This game is one of many simple games created by Google that are AI based as part of a project known as 'A.I. Experiments'.
# Python Implementation
Network Used- Convolutional Neural Network
# Setup 

1. Get the dataset as mentioned above and place the .npy files in /data folder.
2. Run LoadData.py to load the dataset and store the values train.
3. Now you have, you can run Model_train.py to train the model, you can see the Model, Inference,.... and after load data from LoadData.py, the training process begins.
4. After training, model saved in model/QuickDraw.h5.
5. Run QuickDraw_App.py to use window draw and get what you have drawn.
6. For altering the model, check Model_train.py.

Note: Them Full_source.py is a full file of model, if you don't want to run many files, try run it. 

# Installation
To run this project, you will need to install the following dependencies:  
Python 3.6 or higher  
NumPy  
OpenCV  
Scikit-Learn  
TensorFlow  

# Explained 
• Architecture

• Data Pre-Processing and Data Manipulation

• Activation Functions ( RELU and Sigmoid )

• Convolutional Neural Network

• Max Pooling

• Fully Connected Layers

• Flaterning

Note - If Computational power is not enough to train the model on 10-12 Doodled images or .npy files. Download .npy Files in [Here](https://bit.ly/2P7TDut)  
Quickdraw API for accessing the Quick Draw data that downloads the data files as and when needed, caches them locally and interprets them so they can be used. API https://bit.ly/2DkRCbz
