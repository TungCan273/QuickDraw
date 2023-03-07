import pickle
import numpy as np
from tensorflow.python.keras import layers
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical, np_utils

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow import keras 
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from random import randint
from LoadData import load_data

def get_available_gpus():
  from tensorflow.python.client import device_lib
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

x_train, y_train, x_test, y_test, class_names = load_data('data')
num_classes = 100
image_size = 28

#print random img with class names
def print_img():
  bg,cg = plt.subplots(5,2)
  bg.set_size_inches(11,11)
  for i in range(5):
    for j in range(2):
      id = randint(0,len(x_train))
      cg[i,j].imshow(x_train[id].reshape(28,28))
      cg[i,j].set_title(class_names[int(y_train[id].item())])
  plt.tight_layout()

# Reshape and normalize
x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1).astype('float32')

x_train /= 255.0
x_test /= 255.0

# Convert class vectors to class matrices
y_train = keras.utils.to_categorical(y_train, num_classes) 
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define model
model = keras.Sequential()
model.add(Conv2D(16, (3, 3),
                        padding='same',
                        input_shape=x_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same', activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation= 'relu'))
model.add(MaxPooling2D(pool_size =(2,2)))
model.add(Flatten())
model.add(Dropout(0.6))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(100, activation='softmax'))

# Train model
adam = tf.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['top_k_categorical_accuracy'])
print(model.summary())

def print_model():
   tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_names=True,
  )

#train model 
History = model.fit(x = x_train, y = y_train, validation_data=(x_test,y_test), batch_size = 256, verbose=1, epochs=10)

test_loss, score = model.evaluate(x_test, y_test)
print('loss: ',test_loss)
print('Test accuarcy: {:0.2f}%'.format(score* 100))

def render_training_history(training_history):
    loss = training_history.history['loss']
    val_loss = training_history.history['val_loss']

    accuracy = training_history.history['accuracy']
    val_accuracy = training_history.history['val_accuracy']

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'])

    plt.subplot(1, 2, 2)
    plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
    plt.plot(History.history['top_k_categorical_accuracy'])
    plt.plot(History.history['val_top_k_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'])

    plt.show()


def Inference():
  idx = randint(0, len(x_test))
  img = x_test[idx]
  plt.imshow(img.squeeze()) 
  pred = model.predict(np.expand_dims(img, axis=0))[0]
  ind = (-pred).argsort()[:5]
  print([class_names[x] for x in ind])

def main():
  get_available_gpus()
  print_img()
  print_model()
  render_training_history(History)
  Inference()

main()
#save model
model.save('model\QuickDraw.h5')
