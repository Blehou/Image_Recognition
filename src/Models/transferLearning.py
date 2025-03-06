import tensorflow as tf

def create_MobileNet_model(inputShape: tuple = (224, 224, 3), num_classes: int = 36) -> tf.keras.Model:

    # Load the pre-trained model without the original fully connected layers
    Mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=inputShape)

    # Freeze base layers
    Mobilenet.trainable = False

    # Build a sequential model
    model = tf.keras.Sequential([
        Mobilenet,  # Pre-trained CNN part
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),  
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    return model