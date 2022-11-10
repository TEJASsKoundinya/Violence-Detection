import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

m = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4")])
m.build([None, 224, 224, 3])  # Batch input shape.
m.save("my_new_model") # defaults to save as SavedModel in tensorflow 2

New_Model = tf.keras.models.load_model('saved_model.pb') # Loading the Tensorflow Saved Model (PB)
print(New_Model.summary())
