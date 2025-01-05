import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
import os

training_set = train_datagen.flow_from_directory(  # automatic shit for convert to numpy.
    os.path.expanduser("~Desktop/python/CNN/dataset/training_set"),
    target_size = (64,64),
    batch_size = 32,    #   (64, 64, 3) → (32, 64, 64, 3) ingeed urd ni batch_size -iig oruulad ugnu.
    class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(
    os.path.expanduser("~Desktop/python/CNN/dataset/test_set"),
    target_size = (64,64),
    batch_size = 32,
    class_mode = "binary")

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu",input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128,activation="relu"))
cnn.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
cnn.compile(optimizer='adam',loss="binary_crossentropy",metrics = ['accuracy'])
cnn.fit(x=training_set,validation_data=test_set,epochs=25)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img("dataset/test_image_1.jpg",target_size=(64,64))
test_image = image.img_to_array()
test_image = np.expand_dims(test_image,axis=0)
  # The CNN model expects a batch of images as input, even if it’s just one image.The added dimension represents the batch size, making it compatible with the model’s input requirements.
  # (64, 64, 3) → (1, 64, 64, 3)  iimerduu bolj uurchlugduj baigaa.
result = cnn.predict(test_image)

training_set.class_indices  # {'cats': 0, 'dogs': 1} iim baidliin -- dictionary mapping class names to integer labels. hiij ugnuuu.
if result[0][0] > 0.5:   # ehnii teg ni bol first image in the batch,second 0 is for probability value for the predicted class.
    prediction = "dog"
else :
    prediction = "cat"
print(prediction)

