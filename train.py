from keras.models import Sequential
from keras.layers import Dropout, Dense
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM
import tensorflowjs as tfjs
import tensorflow as tf
import numpy as np
import os
import cv2

batch_size = 128

tf.compat.v1.disable_eager_execution()

def bring_data_from_directory():
  datagen = ImageDataGenerator(rescale=1. / 255)
  train_generator = datagen.flow_from_directory(
          'Dataset/Images/Train',
          target_size=(160,120),
          batch_size=batch_size,
          class_mode='categorical',  # this means our generator will only yield batches of data, no labels
          shuffle=True,
          classes=['Anomaly','Normal'])

  validation_generator = datagen.flow_from_directory(
          'Dataset/Images/Test',
          target_size=(160,120),
          batch_size=batch_size,
          class_mode='categorical',  # this means our generator will only yield batches of data, no labels
          shuffle=True,
          classes=['Anomaly','Normal'])
  return train_generator,validation_generator

def load_VGG16_model():
  base_model = VGG16(weights='imagenet', include_top=False, input_shape=(160,120,3))
  print("Model loaded..!")
  print(base_model.summary())
#  tfjs.converters.save_keras_model(base_model,"VGG16-top")
  return base_model

def extract_features(train_generator,validation_generator,base_model):
  train_data = np.load('video_x_VGG16.npy')
  train_labels = np.load('video_y_VGG16.npy')
  train_data,train_labels = shuffle(train_data,train_labels)
  validation_data = np.load('video_x_validate_VGG16.npy')
  validation_labels = np.load('video_y_validate_VGG16.npy')
  validation_data,validation_labels = shuffle(validation_data,validation_labels)

  train_data = train_data.reshape(train_data.shape[0],
                     train_data.shape[1] * train_data.shape[2],
                     train_data.shape[3])
  validation_data = validation_data.reshape(validation_data.shape[0],
                     validation_data.shape[1] * validation_data.shape[2],
                     validation_data.shape[3])
  
  return train_data,train_labels,validation_data,validation_labels

def train_model(train_data,train_labels,validation_data,validation_labels):
  model = Sequential()
  model.add(LSTM(256,dropout=0.2,input_shape=(train_data.shape[1],
                     train_data.shape[2])))
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(128,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(16,activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(2, activation='softmax'))
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('video_4_LSTM_1_1024_64(25).h5', monitor='val_loss', save_best_only=True, verbose=0) ]
  nb_epoch = 25
  model.fit(train_data,train_labels,validation_data=(validation_data,validation_labels),batch_size=batch_size,epochs=nb_epoch,callbacks=callbacks,shuffle=True,verbose=1)
  model.save('Model_V(25).h5')
  tfjs.converters.save_keras_model(model,'Prediction_Model_1')
  return model

def test_on_whole_videos(train_data,train_labels,validation_data,validation_labels):
  parent = os.listdir("./video/test")
  x = []
  y = []
  count = 0
  output = 0
  count_video = 0
  correct_video = 0
  total_video = 0
  base_model = load_VGG16_model()
  model = train_model(train_data,train_labels,validation_data,validation_labels)
  for video_class in parent[1:]:
      print(video_class)
      child = os.listdir("./video/test" + "/" + video_class)
      for class_i in child[1:]:
          sub_child = os.listdir("./video/test" + "/" + video_class + "/" + class_i)
          for image_fol in sub_child[1:]:
              if (video_class ==  'class_4' ):
                  if(count%4 == 0):
                      image = cv2.imread("./video/test" + "/" + video_class + "/" + class_i + "/" + image_fol)
                      image = cv2.resize(image , (160,120))

                      x.append(image)
                      y.append(output)
                  count+=1

              else:
                  if(count%4 == 0):
                      image = cv2.imread("./video/test" + "/" + video_class + "/" + class_i + "/" + image_fol)
                      image = cv2.resize(image , (160,120))
                      x.append(image)
                      y.append(output)
                  count+=1
          x = np.array(x)
          y = np.array(y)
          x_features = base_model.predict(x)

          correct = 0
          
          answer = model.predict(x_features)
          for i in range(len(answer)):
              if(y[i] == np.argmax(answer[i])):
                  correct+=1
          print(correct,"correct",len(answer))
          total_video+=1
          if(correct>= len(answer)/2):
              correct_video+=1
          x = []
          y = []
          count_video+=1
      output+=1

  print ("correct_video",correct_video,"total_video",total_video)
  print ("The accuracy for video classification of ",total_video, " videos is ", (correct_video/total_video))
  
if __name__ == '__main__':
  train_generator,validation_generator = bring_data_from_directory()
  base_model = load_VGG16_model() 
  train_data,train_labels,validation_data,validation_labels = extract_features(train_generator,validation_generator,base_model)
  train_model(train_data,train_labels,validation_data,validation_labels)
  test_on_whole_videos(train_data,train_labels,validation_data,validation_labels)
