import tensorflow as tf


def build_model(input_shape=(227, 227, 3), num_classes=1000, dropout_rate=0.5):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='valid',
                               activation=tf.keras.activations.relu)(inputs)

    # x = tf.nn.local_response_normalization(input=x, depth_radius=5, bias=2, alpha=10 ** -4, beta=0.75)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same',
                               activation=tf.keras.activations.relu)(x)

    # x = tf.nn.local_response_normalization(input=x, depth_radius=5, bias=2, alpha=10 ** -4, beta=0.75)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same',
                               activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same',
                               activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',
                               activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=4096, activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    x = tf.keras.layers.Dense(units=4096, activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    outputs = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.layers.Softmax())(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model
