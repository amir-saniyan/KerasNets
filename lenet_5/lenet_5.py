import tensorflow as tf


def build_model(input_shape=(32, 32, 1), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='valid',
                               activation=tf.keras.activations.relu)(inputs)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='valid',
                               activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=120, activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dense(units=84, activation=tf.keras.activations.relu)(x)

    outputs = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.layers.Softmax())(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model
