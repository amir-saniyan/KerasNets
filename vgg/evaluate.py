import argparse

import tensorflow as tf

from dataset import get_test_dataset


def main():
    parser = argparse.ArgumentParser(description='Evaluates the model')

    parser.add_argument('--test-directory', type=str, required=False, default='../datasets/CIFAR-10/test',
                        help='Directory where the test data is located')
    parser.add_argument('--batch-size', type=int, required=False, default=256, help='Size of the batches of data')
    parser.add_argument('--model-path', type=str, required=False, default='./checkpoint', help='Path to load the model')

    args = parser.parse_args()

    test_directory = args.test_directory
    batch_size = args.batch_size
    model_path = args.model_path

    model = tf.keras.models.load_model(filepath=model_path, compile=True)

    input_shape = model.input_shape[1:]
    input_image_size = input_shape[:-1]

    test_dataset = get_test_dataset(directory=test_directory, batch_size=batch_size, image_size=input_image_size)

    model.evaluate(test_dataset)

    print('Evaluation done successfully')


if __name__ == '__main__':
    main()
