import tensorflow as tf


def build_model(configuration='vgg19', input_shape=(224, 224, 3), num_classes=1000, dropout_rate=0.5):
    if configuration not in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
        raise Exception('Configuration should be one of these values: vgg11, vgg13, vgg16, vgg19')

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same',
                               activation=tf.keras.activations.relu)(inputs)

    if configuration == 'vgg13' or configuration == 'vgg16' or configuration == 'vgg19':
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same',
                                   activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same',
                               activation=tf.keras.activations.relu)(x)

    if configuration == 'vgg13' or configuration == 'vgg16' or configuration == 'vgg19':
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same',
                                   activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',
                               activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',
                               activation=tf.keras.activations.relu)(x)

    if configuration == 'vgg16' or configuration == 'vgg19':
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',
                                   activation=tf.keras.activations.relu)(x)

        if configuration == 'vgg19':
            x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',
                                       activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',
                               activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',
                               activation=tf.keras.activations.relu)(x)

    if configuration == 'vgg16' or configuration == 'vgg19':
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',
                                   activation=tf.keras.activations.relu)(x)

        if configuration == 'vgg19':
            x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',
                                       activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',
                               activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',
                               activation=tf.keras.activations.relu)(x)

    if configuration == 'vgg16' or configuration == 'vgg19':
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',
                                   activation=tf.keras.activations.relu)(x)

        if configuration == 'vgg19':
            x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',
                                       activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=4096, activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    x = tf.keras.layers.Dense(units=4096, activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    outputs = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.layers.Softmax())(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model
