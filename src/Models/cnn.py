import tensorflow as tf


def create_CNN_model(inputShape: tuple = (224, 224, 3), num_classes: int = 36) -> tf.keras.Model:
    """
    Creates a Convolutional Neural Network (CNN) model for image classification.

    Args:
        inputShape (tuple, optional): Shape of the input image (height, width, channels). 
                                      Defaults to (224, 224, 3).
        num_classes (int, optional): Number of output classes. Defaults to 36.

    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=inputShape),
        tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=200, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    return model