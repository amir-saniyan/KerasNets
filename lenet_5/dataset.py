import tensorflow as tf


def get_training_dataset(directory, batch_size=32, image_size=(32, 32), shuffle=True, seed=0, validation_split=0.2):
    training_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset='training')

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset='validation')

    return training_dataset, validation_dataset


def get_test_dataset(directory, batch_size=32, image_size=(32, 32)):
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False)

    return test_dataset
