import tensorflow as tf
# Converting a SavedModel to a TensorFlow Lite model.
saved_model_dir = r"ViolenceDetection/movnet/saved_model.pb"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
