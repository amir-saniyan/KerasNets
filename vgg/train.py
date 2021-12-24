import argparse

import tensorflow as tf

import vgg
from dataset import get_training_dataset


def main():
    parser = argparse.ArgumentParser(description='Trains the model')

    parser.add_argument('--configuration', type=str, required=False, default='vgg19',
                        help='Model configuration, one of: vgg11, vgg13, vgg16, vgg19')
    parser.add_argument('--train-directory', type=str, required=False, default='../datasets/CIFAR-10/train',
                        help='Directory where the train data is located')
    parser.add_argument('--batch-size', type=int, required=False, default=256, help='Size of the batches of data')
    parser.add_argument('--input-image-width', type=int, required=False, default=32, help='Input image width')
    parser.add_argument('--input-image-height', type=int, required=False, default=32, help='Input image height')
    parser.add_argument('--shuffle', type=lambda x: (str(x).lower() == 'true'), required=False, default=True,
                        help='Whether to shuffle the data')
    parser.add_argument('--seed', type=int, required=False, default=0,
                        help='Random seed for shuffling and transformations')
    parser.add_argument('--dropout-rate', type=float, required=False, default=0.5,
                        help='Fraction of the input units to drop')
    parser.add_argument('--learning-rate', type=float, required=False, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, required=False, default=0.9,
                        help='Hyper-parameter that accelerates gradient descent in the relevant direction and dampens '
                             'oscillations')
    parser.add_argument('--epochs', type=int, required=False, default=50, help='Number of epochs')
    parser.add_argument('--validation_split', type=float, required=False, default=0.20,
                        help='Fraction of the training data to be used as validation data')
    parser.add_argument('--logs-path', type=str, required=False, default='./logs',
                        help='Path of the directory where to save the log files to be parsed by TensorBoard')
    parser.add_argument('--model-path', type=str, required=False, default='./checkpoint', help='Path to save the model')

    args = parser.parse_args()

    configuration = args.configuration
    train_directory = args.train_directory
    batch_size = args.batch_size
    input_image_size = (args.input_image_width, args.input_image_height)
    input_shape = (*input_image_size, 3)
    shuffle = args.shuffle
    seed = args.seed
    dropout_rate = args.dropout_rate
    learning_rate = args.learning_rate
    momentum = args.momentum
    epochs = args.epochs
    validation_split = args.validation_split
    logs_path = args.logs_path
    model_path = args.model_path

    training_dataset, validation_dataset = get_training_dataset(directory=train_directory, batch_size=batch_size,
                                                                image_size=input_image_size, shuffle=shuffle, seed=seed,
                                                                validation_split=validation_split)

    num_classes = len(training_dataset.class_names)

    model = vgg.build_model(configuration=configuration, input_shape=input_shape, num_classes=num_classes,
                            dropout_rate=dropout_rate)

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    loss = tf.keras.losses.CategoricalCrossentropy()

    categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()
    metrics = [categorical_accuracy]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.summary()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1,
                                                                   save_best_only=True)
    callbacks = [tensorboard_callback, model_checkpoint_callback]

    model.fit(training_dataset, epochs=epochs, validation_data=validation_dataset, callbacks=callbacks)

    print('Training done successfully')


if __name__ == '__main__':
    main()
