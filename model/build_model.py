"""
build_model.py
---------------
Defines the CNN model using transfer learning (MobileNetV2 pretrained
on ImageNet). MobileNetV2 is chosen because it is lightweight and fast
to train on a laptop/free-tier GPU (Colab), compared to ResNet50.

To switch to ResNet50 instead, just change the `base_model` line
(marked below) - the rest of the code stays the same.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(num_classes, img_size=(224, 224), fine_tune=False):
    """
    Builds a transfer-learning model.

    Args:
        num_classes: number of output disease classes (38 for PlantVillage)
        img_size: input image size, e.g. (224, 224)
        fine_tune: if True, unfreezes the last few layers of the base
                   model for fine-tuning (use this in a second training
                   phase after the top layers have already converged)
    """
    input_shape = img_size + (3,)

    # ----------------------------------------------------------
    # Base model: MobileNetV2 pretrained on ImageNet, no top layer
    # (To use ResNet50 instead, replace this block with:
    #   base_model = tf.keras.applications.ResNet50(
    #       input_shape=input_shape, include_top=False, weights="imagenet")
    #   and use tf.keras.applications.resnet50.preprocess_input below)
    # ----------------------------------------------------------
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = fine_tune
    if fine_tune:
        # Only unfreeze the last 30 layers to avoid destroying learned features
        for layer in base_model.layers[:-30]:
            layer.trainable = False

    # Preprocessing expected by MobileNetV2 (scales pixels to [-1, 1])
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = layers.Input(shape=input_shape)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)                 # reduces overfitting
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="plant_disease_mobilenetv2")
    return model
