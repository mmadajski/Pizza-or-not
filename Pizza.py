import glob

import keras
import numpy as np
import cv2 as cv2
import tensorflow as tf
from keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, Dense, Activation, BatchNormalization, ReLU, \
    AveragePooling2D
from random import seed
from random import choices
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

paths_pizza = glob.glob(".//pizza_not_pizza//pizza//*.jpg")
paths_not_pizza = glob.glob(".//pizza_not_pizza//not_pizza//*.jpg")

seed(123)
selected_pizza_train = choices(paths_pizza, k=int(len(paths_pizza) * 0.7))

seed(456)
selected_not_pizza_train = choices(paths_not_pizza, k=int(len(paths_not_pizza) * 0.7))
test_paths_pizza = list(set(paths_pizza) - set(selected_pizza_train))
test_paths_not_pizza = list(set(paths_not_pizza) - set(selected_not_pizza_train))

test_answers = []

data_train_list = []
data_train_answer = []

for i in range(len(selected_pizza_train)):
    img = cv2.imread(selected_pizza_train[i])
    img_resize = cv2.resize(img, (256, 256))
    data_train_list.append(img_resize)
    data_train_answer.append(1)

    # Simple data augmentation by flipping images.
    image_flip = cv2.flip(img_resize, 1)
    data_train_list.append(image_flip)
    data_train_answer.append(1)

    img = cv2.imread(selected_not_pizza_train[i])
    img_resize = cv2.resize(img, (256, 256))
    data_train_list.append(img_resize)
    data_train_answer.append(0)

    # Same data augmentation for 0 class.
    image_flip = cv2.flip(img_resize, 1)
    data_train_list.append(image_flip)
    data_train_answer.append(0)

data_test = []
data_test_answers = []

for i in range(len(test_paths_pizza)):
    img = cv2.imread(test_paths_pizza[i])
    img_resize = cv2.resize(img, (256, 256))
    data_test.append(img_resize)
    data_test_answers.append(1)

for i in range(len(test_paths_not_pizza)):
    img = cv2.imread(test_paths_not_pizza[i])
    img_resize = cv2.resize(img, (256, 256))
    data_test.append(img_resize)
    data_test_answers.append(0)

train_data_np = np.array(data_train_list)
train_answers_np = np.array(data_train_answer)

test_data_np = np.array(data_test)
test_answers_np = np.array(data_test_answers)

CNN = tf.keras.Sequential()

CNN.add(InputLayer(input_shape=(256, 256, 3)))
CNN.add(Conv2D(16, (5, 5), activation="relu"))
CNN.add(MaxPool2D(5, 5))
CNN.add(Conv2D(32, (5, 5), activation="relu"))
CNN.add(MaxPool2D(5, 5))
CNN.add(Conv2D(64, (5, 5), activation="relu"))
CNN.add(MaxPool2D(5, 5))

CNN.add(Flatten())
CNN.add(Dense(64, activation="relu"))
CNN.add(Dense(32, activation="relu"))
CNN.add(Dense(16, activation="relu"))
CNN.add(Dense(1))
CNN.add(Activation("sigmoid"))

CNN.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy', 'Recall'])


def create_res_net_block(x, filters_num, filter_size, layers_num):
    x_next = x

    for layer in range(layers_num - 1):
        x_next = Conv2D(filters_num, filter_size, padding="same")(x_next)
        x_next = BatchNormalization()(x_next)
        x_next = ReLU()(x_next)

    # Last iteration
    x_next = Conv2D(x.shape[-1], filter_size, padding="same")(x_next)
    x_next = BatchNormalization()(x_next)

    x = tf.keras.layers.Add()([x, x_next])
    x = tf.keras.layers.ReLU()(x)

    return x


# ResNet model
ResNet_input = tf.keras.Input(shape=(256, 256, 3))
next_layer = Conv2D(16, (5, 5))(ResNet_input)
next_layer = BatchNormalization()(next_layer)
next_layer = ReLU()(next_layer)
next_layer = MaxPool2D((2, 2))(next_layer)

next_layer = create_res_net_block(next_layer, 32, (5, 5), 3)
next_layer = MaxPool2D((2, 2))(next_layer)
next_layer = create_res_net_block(next_layer, 64, (5, 5), 3)

next_layer = AveragePooling2D((2, 2))(next_layer)
next_layer = Flatten(data_format=None)(next_layer)

next_layer = Dense(128, activation="relu")(next_layer)
next_layer = Dense(1, activation="sigmoid")(next_layer)

ResNet = tf.keras.Model(inputs=ResNet_input, outputs=next_layer)
ResNet.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy', 'Recall'])

history_cnn = CNN.fit(train_data_np, train_answers_np, batch_size=45, epochs=10)
history_res = ResNet.fit(train_data_np, train_answers_np, batch_size=45, epochs=10)

x = [i for i in range(len(history_cnn.history["loss"]))]
plt.plot(x, history_cnn.history["loss"], label="CNN loss")
plt.plot(x, history_res.history["loss"], label="ResNet loss")
plt.legend(["CNN loss", "ResNet loss"])
plt.xlabel("Epochs")
plt.savefig('history_loss.png')
plt.clf()

plt.plot(x, history_cnn.history["accuracy"], label="CNN accuracy")
plt.plot(x, history_res.history["accuracy"], label="ResNet accuracy")
plt.legend(["CNN accuracy", "ResNet accuracy"])
plt.xlabel("Epochs")
plt.savefig('history_accuracy.png')

train_pred_cnn = CNN.predict(train_data_np)
pred_train_cnn_list = [int(i[0] > 0.5) for i in train_pred_cnn]
confusion_matrix_cnn = tf.math.confusion_matrix(train_answers_np, pred_train_cnn_list)

print("Train data CNN params: ")
print(f"Accuracy: %.2f" % tf.get_static_value((confusion_matrix_cnn[0][0] + confusion_matrix_cnn[1][1]) / np.sum(confusion_matrix_cnn)))
print(f"Recall\\Sensitivity: %.2f" % tf.get_static_value(confusion_matrix_cnn[1][1] / (confusion_matrix_cnn[1][1] + confusion_matrix_cnn[0][1])))
print(f"Specificity: %.2f" % tf.get_static_value(confusion_matrix_cnn[0][0] / (confusion_matrix_cnn[0][0] + confusion_matrix_cnn[1][0])))
print("----------------------------")

test_pred_cnn = CNN.predict(test_data_np)
pred_test_cnn_list = [int(i[0] > 0.5) for i in test_pred_cnn]

confusion_matrix_cnn = tf.math.confusion_matrix(test_answers_np, pred_test_cnn_list)
print("Test data CNN params: ")
print(f"Accuracy: %.2f" % tf.get_static_value((confusion_matrix_cnn[0][0] + confusion_matrix_cnn[1][1]) / np.sum(confusion_matrix_cnn)))
print(f"Recall\\Sensitivity: %.2f" % tf.get_static_value(confusion_matrix_cnn[1][1] / (confusion_matrix_cnn[1][1] + confusion_matrix_cnn[0][1])))
print(f"Specificity: %.2f" % tf.get_static_value(confusion_matrix_cnn[0][0] / (confusion_matrix_cnn[0][0] + confusion_matrix_cnn[1][0])))
print("----------------------------")


train_pred_res = ResNet.predict(train_data_np)
pred_train_res_list = [int(i[0] > 0.5) for i in train_pred_res]
confusion_matrix_res = tf.math.confusion_matrix(train_answers_np, pred_train_res_list)

print("Train data ResNet params: ")
print(f"Accuracy: %.2f" % tf.get_static_value((confusion_matrix_res[0][0] + confusion_matrix_res[1][1]) / np.sum(confusion_matrix_res)))
print(f"Recall\\Sensitivity: %.2f" % tf.get_static_value(confusion_matrix_res[1][1] / (confusion_matrix_res[1][1] + confusion_matrix_res[0][1])))
print(f"Specificity: %.2f" % tf.get_static_value(confusion_matrix_res[0][0] / (confusion_matrix_res[0][0] + confusion_matrix_res[1][0])))
print("----------------------------")

test_pred_res = ResNet.predict(test_data_np)
pred_test_res_list = [int(i[0] > 0.5) for i in test_pred_res]

confusion_matrix_res = tf.math.confusion_matrix(test_answers_np, pred_test_res_list)
print("Test data ResNet params: ")
print(f"Accuracy: %.2f" % tf.get_static_value((confusion_matrix_res[0][0] + confusion_matrix_res[1][1]) / np.sum(confusion_matrix_res)))
print(f"Recall\\Sensitivity: %.2f" % tf.get_static_value(confusion_matrix_res[1][1] / (confusion_matrix_res[1][1] + confusion_matrix_res[0][1])))
print(f"Specificity: %.2f" % tf.get_static_value(confusion_matrix_res[0][0] / (confusion_matrix_res[0][0] + confusion_matrix_res[1][0])))
print("----------------------------")
