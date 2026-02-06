import tensorflow as tf
import numpy as np

def load_client_data(client_id, num_clients=5):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()

    classes_per_client = 2
    start_class = client_id * classes_per_client
    end_class = start_class + classes_per_client

    mask = np.isin(y_train, list(range(start_class, end_class)))
    x_train, y_train = x_train[mask], y_train[mask]

    return (x_train, y_train), (x_test, y_test)

