import tensorflow as tf

def efficient_net_2s(input_layer):
    efficient_net_2s = tf.keras.applications.efficientnet_v2.EfficientNetV2S(weights = "imagenet", include_top = False, input_tensor = input_layer)
    return tf.keras.layers.Flatten(name='flatten')(efficient_net_2s.output)

def efficient_net_2m(input_layer):
    efficientNetV2M = tf.keras.applications.efficientnet_v2.EfficientNetV2M(weights = "imagenet", include_top = False, input_tensor = input_layer)
    return tf.keras.layers.Flatten(name='flatten')(efficientNetV2M.output)

def buil_VGG19(input_layer):
    layers = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_1')(input_layer)
    layers = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_2')(layers)
    layers = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max_pooling2d_1')(layers)
    
    layers = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_1')(layers)
    layers = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_2')(layers)
    layers = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max_pooling2d_2')(layers)

    layers = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_1')(layers)
    layers = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_2')(layers)
    layers = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_3')(layers)
    layers = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max_pooling2d_3')(layers)

    layers = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_1')(layers)
    layers = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_2')(layers)
    layers = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_3')(layers)
    layers = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max_pooling2d_4')(layers)

    layers = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_1')(layers)
    layers = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_2')(layers)
    layers = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_3')(layers)
    layers = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max_pooling2d_5')(layers)

    return tf.keras.layers.Flatten(name='flatten')(layers)

def build_model(n_classes, shape=(640,640,3)):
    input_layer = tf.keras.layers.Input(shape)

    base_layers = efficient_net_2s(input_layer)

    softmaxHead = tf.keras.layers.Dense(512, activation="relu", name='dense_s1')(base_layers)
    softmaxHead = tf.keras.layers.Dropout(0.5)(softmaxHead)
    softmaxHead = tf.keras.layers.Dense(512, activation="relu", name='dense_s3')(softmaxHead)
    softmaxHead = tf.keras.layers.Dropout(0.5)(softmaxHead)
    softmaxHead = tf.keras.layers.Dense(n_classes, activation="softmax", name="class_label")(softmaxHead)

    bboxHead = tf.keras.layers.Dense(128, activation="relu", name='dense_b2')(base_layers)
    bboxHead = tf.keras.layers.Dense(64, activation="relu", name='dense_b3')(bboxHead)
    bboxHead = tf.keras.layers.Dense(32, activation="relu", name='dense_b4')(bboxHead)
    bboxHead = tf.keras.layers.Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)  

    return tf.keras.models.Model(inputs=input_layer, outputs=(bboxHead, softmaxHead))

