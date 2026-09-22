"""
data_utils.py
-------------
Helper functions to load the split dataset (train/val/test folders) into
tf.data pipelines, with data augmentation applied to the training set.
"""

import tensorflow as tf

IMG_SIZE = (224, 224)   # required input size for MobileNetV2 / ResNet50
BATCH_SIZE = 32


def get_datasets(data_dir="data/split", img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Loads train/val/test datasets from folders using Keras' image_dataset_from_directory.
    Expects the folder structure:
        data_dir/train/<class_name>/*.jpg
        data_dir/val/<class_name>/*.jpg
        data_dir/test/<class_name>/*.jpg
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",  # one-hot labels, needed for softmax + categorical_crossentropy
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names  # save BEFORE prefetch/map for later use

    # --------------------------------------------------------------
    # Data augmentation layers (only applied to training data)
    # --------------------------------------------------------------
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),      # +/- 20% of 360 degrees
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")

    def augment(images, labels):
        images = data_augmentation(images, training=True)
        return images, labels

    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Performance: cache + prefetch
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
