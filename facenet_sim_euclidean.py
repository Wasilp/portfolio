# face detection
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.models import model_from_json
from keras.preprocessing import image
from sklearn.preprocessing import normalize
import keras.backend as K
import numpy as np
import cv2

"""
face_api class 
function available:preprocess_image
                    findEuclideanDistance
                    verifyFace 
Generate routine to find differences between 2 img of persons
"""

class face_api:
    def __init__(self, file, metric="euclidean"):
        self.file = file
        self.metric = metric
        self.euc_treshold = 0.35

    # extract a single face from a given photograph
    def preprocess_image(self, filename, required_size=(160, 160)):
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert("RGB")
        # convert to array
        pixels = asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]["box"]

        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = preprocess_input(face_array)
        return face_array

        """findEuclideanDistance find euclidean distance between two sources
            return np array 
        """

    def findEuclideanDistance(self, source_representation, test_representation):
        return np.linalg.norm(source_representation - test_representation)

        """VerifyFace the prediction from two sources
           return: str,float 
        """

    def verifyFace(self, file_uploaded, model):
        # produce 128-dimensional representation
        img1 = model.predict(self.preprocess_image(self.file))
        K.clear_session()
        img2 = model.predict(self.preprocess_image(file_uploaded))
        K.clear_session()

        if self.metric == "euclidean":
            img1 = normalize(img1, norm="l2")
            img2 = normalize(img2, norm="l2")

            euclidean_distance = self.findEuclideanDistance(img1, img2)
            nope = "Too bad you don't look like me, we know you wish right ? You still can hire me to be part of your team..just sayin."
            pogger = "Wow! What an awesome person,you should hire him definitely"
            if euclidean_distance < self.euc_treshold:
                return pogger, euclidean_distance
            else:
                return nope, euclidean_distance

