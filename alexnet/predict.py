import argparse

import cv2
import numpy as np
import tensorflow as tf

CIFAR_10_LABELS = { 0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                    8: 'ship', 9: 'truck'}


def main():
    parser = argparse.ArgumentParser(description='Predicts the model')

    parser.add_argument('--image-path', type=str, required=True, help='Image path to predict')
    parser.add_argument('--model-path', type=str, required=False, default='./checkpoint', help='Path to load the model')

    args = parser.parse_args()

    image_path = args.image_path
    model_path = args.model_path

    model = tf.keras.models.load_model(filepath=model_path, compile=False)

    input_shape = model.input_shape[1:]
    input_image_size = input_shape[:-1]

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, input_image_size)
    image = image.astype(np.float32)
    image = image.reshape((1, *input_shape))

    prediction_result = model.predict(image)

    predicted_index = np.argmax(prediction_result)
    print('Predicted index:', predicted_index)

    if predicted_index in CIFAR_10_LABELS:
        print('Predicted label (CIFAR-10):', CIFAR_10_LABELS[predicted_index])


if __name__ == '__main__':
    main()
