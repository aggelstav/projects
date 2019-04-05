import json

import numpy as np

import tensorflow as tf
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

global graph
graph = tf.get_default_graph()
model = load_model("predictor.h5")



@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    print(type(data))


    with graph.as_default():
        X = [data for _ in range(len(model.input))]
        prediction = model.predict(X)
        output = np.argmax(prediction, axis=1)

        if output[-1] == 1:
            print("up")
        else:
            print("down")

    return jsonify(data)    



if __name__ == '__main__':
    app.run(port=5000, debug=True)
