import numpy as np
import tensorflow as tf
from keras.layers import InputLayer, Conv2D, MaxPool2D,\
    Flatten, Dense, Activation, BatchNormalization, ReLU, \
    AveragePooling2D
from typing import TypeVar
array_like = TypeVar("array_like")

def build_cnn_network(params: dict) -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(InputLayer(input_shape=(256, 256, 3)))

    for layer in range(params["convolutional_layers"]):
        model.add(Conv2D(params["filters_number"][layer], params["kernel_size"], activation="relu"))
        model.add(MaxPool2D(params["polling_size"]))

    model.add(Flatten())

    for layer_neuron_number in params["dense_layers_neurons_num"]:
        model.add(Dense(layer_neuron_number, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    return model


def build_resnet_network(params: dict) -> tf.keras.Sequential:
    model_input = tf.keras.Input(shape=(256, 256, 3))

    next_layer = Conv2D(params["initial_filters"], params["kernel_size"])(model_input)
    next_layer = BatchNormalization()(next_layer)
    next_layer = ReLU()(next_layer)
    next_layer = MaxPool2D(params["polling_size"])(next_layer)

    for residual_block in range(params["residual_blocks"]):
        next_layer = create_res_net_block(next_layer, params["residual_filters_number"][residual_block], params["kernel_size"], params["convolution_layers_inside_res_block"])
        next_layer = MaxPool2D((2, 2))(next_layer)

    next_layer = AveragePooling2D((2, 2))(next_layer)
    next_layer = Flatten()(next_layer)

    for dense_layer_neurons in params["dense_layers_neurons_num"]:
        next_layer = Dense(dense_layer_neurons, activation="relu")(next_layer)

    next_layer = Dense(1, activation="sigmoid")(next_layer)

    model = tf.keras.Model(inputs=model_input, outputs=next_layer)

    return model


def create_res_net_block(x, filters_num, kernel_size, layers_num):
    x_next = x

    for layer in range(layers_num - 1):
        x_next = Conv2D(filters_num, kernel_size, padding="same")(x_next)
        x_next = BatchNormalization()(x_next)
        x_next = ReLU()(x_next)

    # Last iteration
    x_next = Conv2D(x.shape[-1], kernel_size, padding="same")(x_next)
    x_next = BatchNormalization()(x_next)

    x = tf.keras.layers.Add()([x, x_next])
    x = tf.keras.layers.ReLU()(x)
    return x


def calculate_metrics(answers: array_like, predictions: array_like) -> tuple[any]:
    """
    Calculates accuracy, recall, specificity.
    :param answers:
    :param predictions:
    :return:
    """

    confusion_matrix_cnn = tf.math.confusion_matrix(answers, predictions)

    accuracy = tf.get_static_value(
        (confusion_matrix_cnn[0][0] + confusion_matrix_cnn[1][1]) / np.sum(confusion_matrix_cnn))
    recall = tf.get_static_value(
        confusion_matrix_cnn[1][1] / (confusion_matrix_cnn[1][1] + confusion_matrix_cnn[0][1]))
    specificity = tf.get_static_value(
        confusion_matrix_cnn[0][0] / (confusion_matrix_cnn[0][0] + confusion_matrix_cnn[1][0]))

    return accuracy, recall, specificity
