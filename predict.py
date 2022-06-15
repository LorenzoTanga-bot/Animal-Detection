import pickle
from pathlib import Path
import os

import cv2
import imutils
import numpy as np
import tensorflow as tf

import config

def _predict(image, imagePath, model, label_binarizer):
    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]

    i = np.argmax(labelPreds, axis=1)
    label = label_binarizer.classes_[i][0]

    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=640)
    (h, w) = image.shape[:2]

    xmin = int(startX * w)
    ymin = int(startY * h)
    xmax = int(endX * w)
    ymax = int(endY*w)

    return image, label, xmin, ymin, xmax, ymax


def predict(image_paths, output_path, model_path):
    model = tf.keras.models.load_model(model_path)
    label_binarizer = pickle.loads(open(config.paths["LABEL_BINARIZER"] + "lb.pickle", "rb").read())

    for image_path in image_paths:
        image = tf.keras.preprocessing.image.img_to_array(
            tf.keras.preprocessing.image.load_img(
            image_path, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))) / 255.0
        image = np.expand_dims(image, axis=0)

        image, label, xmin, ymin, xmax, ymax = _predict(image, image_path, model, label_binarizer)

        y = ymin - 10 if ymin - 10 > 10 else ymin + 10
        cv2.putText(image, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        cv2.imwrite(output_path + Path(image_path).stem + ".png", image)