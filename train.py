import os
import pickle
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import config
from model import build_model
from predict import predict


def save_plot(history_training):
    print("\n[INFO] Saving model accuracy/loss\n")

    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    plt.plot(history_training.history['bounding_box_accuracy'])
    plt.plot(history_training.history['class_label_accuracy'])
    plt.plot(history_training.history['val_bounding_box_accuracy'])
    plt.plot(history_training.history['val_class_label_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['bbox', 'clabel', 'val_bbox', 'val_clabel'], loc='upper center',
               bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    plt.savefig(config.paths["HISTORY_TRAINING"] +
                "model_accuracy_" + time_now + ".png")

    plt.clf()

    plt.plot(history_training.history['bounding_box_loss'])
    plt.plot(history_training.history['class_label_loss'])
    plt.plot(history_training.history['val_bounding_box_loss'])
    plt.plot(history_training.history['val_class_label_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['bbox', 'clabel', 'val_bbox', 'val_clabel'], loc='upper center',
               bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    plt.savefig(config.paths["HISTORY_TRAINING"] +
                "model_loss_" + time_now + ".png")

    plt.clf()


def _load_images(annotations, path, start, end):
    images = []
    labels = []
    bboxes = []

    for row in annotations[start: end]:
        row = row.split(",")
        (filename, width, height, label, xmin, ymin, xmax, ymax) = row

        width = float(width)
        height = float(height)

        xmin = float(xmin) / width
        ymin = float(ymin) / height
        xmax = float(xmax) / width
        ymax = float(ymax) / height

        image = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(os.path.join(
            path, filename), target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)))

        images.append(image)
        labels.append(label)
        bboxes.append((xmin, ymin, xmax, ymax))

    return images, labels, bboxes


def _save_label_binarizer(label_binarizer):
    print("\n[INFO] Saving label binarizer\n")
    f = open(os.path.join(config.paths["LABEL_BINARIZER"], "lb.pickle"),  "wb")
    f.write(pickle.dumps(label_binarizer))
    f.close()


def load_images(path, index_sub_set, tot_sub_set):
    print("\n[INFO] Loading train images: ",
          index_sub_set, "of", config.SUPER_EPOCHS, "\n")

    annotations = open(path + "_annotations.csv").read().strip().split("\n")
    annotations.pop(0)
    random.shuffle(annotations)
    
    n_parsial_annotations = int(len(annotations) / tot_sub_set)

    images, labels, bboxes = _load_images(
        annotations, path, n_parsial_annotations * index_sub_set,  n_parsial_annotations * (index_sub_set + 1))

    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(np.array(labels))

    if len(label_binarizer.classes_) < config.N_CLASS:
        return load_images(path, index_sub_set, tot_sub_set)

    if not os.path.exists(config.paths["LABEL_BINARIZER"] + "lb.pickle"):
        _save_label_binarizer(label_binarizer)

    images = np.array(images, dtype="float32") / 255.0
    bboxes = np.array(bboxes, dtype="float32")

    return images, labels, bboxes


def train():
    print("\n[INFO] Building model\n")

    model = build_model(
        config.N_CLASS, (config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    model.compile(loss={"class_label": 'categorical_crossentropy',
                  "bounding_box": 'mean_squared_error'}, optimizer='adam', metrics=["accuracy"])

    for index_image in range(0, config.SUPER_EPOCHS):

        image, labels, bbox = load_images(config.paths["DATASET"], index_image, config.SUPER_EPOCHS)

        train_image, test_image, train_labels, test_label, train_bbox, test_bbox = train_test_split(
            image, labels, bbox, test_size=0.20)
        
        train_image, valid_image, train_labels, valid_label, train_bbox, valid_bbox = train_test_split(
            train_image, train_labels, train_bbox, test_size=0.20)

        train_targets = {"class_label": train_labels, "bounding_box": train_bbox}
        
        test_targets = {"class_label": test_label, "bounding_box": test_bbox}
        
        valid_targets = {"class_label": valid_label, "bounding_box": valid_bbox}

        print("\n[INFO] Trainig model\n")
        history_training = model.fit(train_image, train_targets, validation_data=(
            valid_image, valid_targets), batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, shuffle=True, verbose=1)

        model_save_name = config.paths["MODEL"] + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".h5"

        model.save(model_save_name, save_format="h5")

        save_plot(history_training)

    model.evaluate(test_image, test_targets)


    image_test_predict = [os.path.join(config.paths["DATASET_TEST_PREDICT"], f) for f in os.listdir(config.paths["DATASET_TEST_PREDICT"]) if os.path.isfile(os.path.join(config.paths["DATASET_TEST_PREDICT"], f))]  
    predict(image_test_predict, config.paths["DATASET_TEST_PREDICT_OUTPUT"], model_save_name)




if __name__ == "__main__":
    config.setup_paths()
    train()
