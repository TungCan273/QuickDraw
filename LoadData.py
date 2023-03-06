import os
import numpy as np
from sklearn.model_selection import train_test_split

# class_names = ['drums.npy', 'alarmclock.npy', 'apple.npy', 'backpack.npy', 'barn.npy', 
#               'bed.npy', 'bowtie.npy', 'candle.npy', 'door.npy', 'envelope.npy', 
#               'fish.npy', 'guitar.npy', 'icecream.npy', 'mountain.npy', 'star.npy', 
#               'tent.npy', 'toothbrush.npy', 'wristwatch.npy']

#make sure model trained by GPU 
def get_available_gpus():
  from tensorflow.python.client import device_lib
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

folder_path = "C:\\Users\\Acer\\OneDrive\\Máy tính\\Python\\Project AI\\QuickDraw\\data"
all_files = os.listdir(folder_path)

classes = [c.replace('full_numpy_bitmap_', ' ').replace(' ', '') for c in all_files]

#load data
def load_data(root, vfold_ratio=0.2, max_items_per_class=5000):
  data = np.empty([0, 784])
  labels = np.empty([0], dtype=int)
  class_names = []

  for idx, file in enumerate(all_files):
    file = "C:\\Users\\Acer\\OneDrive\\Máy tính\\Python\\Project AI\\QuickDraw\\data\\" + file

    class_data = np.load(file)[0:max_items_per_class, : ]
    class_labels = np.full(class_data.shape[0], idx)

    data = np.concatenate((data, class_data), axis=0)
    labels = np.append(labels, class_labels)

    class_name,_ = os.path.splitext(os.path.basename(file))
    class_names.append(class_name)

  class_data = None
  class_labels = None

  #random dataset 
  x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=42, test_size=0.2)
  x_test = x_test.astype('float32')
  y_test = y_test.astype('float32')
    
  print("x_train\n",x_train, x_train.shape)
  print("y_train\n",y_train, y_train.shape)
  print("x_test\,",x_test, x_test.shape)
  print("y_test\n",y_test, y_test.shape)

  return x_train, y_train, x_test, y_test, class_names