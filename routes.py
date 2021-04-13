from flask import Flask
from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename
from flask_wtf.csrf import CSRFProtect
from flask_cors import CORS, cross_origin
from inception_resnet_v1 import *
from facenet_sim_euclidean import face_api
from glob import glob
import json
import tensorflow as tf
import os


app = Flask(__name__)
app.config["UPLOAD_PATH"] = "static/uploads"
app.config["SECRET_KEY"] = "457895"
csrf = CSRFProtect(app)
CORS(app)


# def load_tflite_model(file):
#     # Load the TFLite model and allocate tensors.
#     interpreter = tf.lite.Interpreter(model_path=file)
#     interpreter.allocate_tensors()
#     return interpreter
# load tfl model
# tfl_file = "./model/facenet.tflite"
# model = load_tflite_model(tfl_file)

model = InceptionResNetV1()
model.load_weights("./model/facenet_weights.h5")


@app.route("/index", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    """
    Display the home page
    """
    return render_template("index.html")


@app.route("/ajax", methods=["POST"])
@cross_origin()
def upload_file():
    if request.method == "POST":
        # get the file from request
        f = request.files.get("file")
        # Saving file and create file_path to display on client
        filename = secure_filename(f.filename)
        file_path = "static/uploads/" + filename
        f.save(os.path.join(app.config["UPLOAD_PATH"], filename))

        # File who will be compared with uploads one
        file = "static/uploads/pw.jpg"
        # Create path from root to send to our model
        file_uploaded =  file_path
        metric = "euclidean"
        face = face_api(file, metric)
        results, euclidean = face.verifyFace(file_uploaded, model)
        print(results, euclidean)

        return jsonify(file_path=file_path, results=results, euclidean=float(euclidean))


if __name__ == "__main__":
    app.run()
