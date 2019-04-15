import tensorflow as tf
import numpy as np

tf.keras.backend.set_learning_phase(0) #ignore dropout at inference
model = tf.keras.models.load_model("./FCN_predictor_128.h5")   
export_path = './TimeSeriesClassifier/1'
input_series = model.input 
output = model.output

with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
    sess,
    export_path,
    inputs={'input_series': input_series},
    outputs={'Prediction': output})
