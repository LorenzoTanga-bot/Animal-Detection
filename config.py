import os

IMAGE_SIZE = 640
N_CLASS = 36
SUPER_EPOCHS = 25
EPOCHS = 100
BATCH_SIZE = 64


paths = {
    "MODEL": "./model/",
    "DATASET": "./dataset/",
    "DATASET_TEST_PREDICT": "./images_test_predict/",
    "DATASET_TEST_PREDICT_OUTPUT": "./images_test_predict/output/",
    "HISTORY_TRAINING": "./history_training/",
    "LABEL_BINARIZER": "./label_binarizer/",
}

def setup_paths():
    for path in paths.values():
        if not os.path.exists(path):
            print("mkdir: ", path)
            os.mkdir(path)
