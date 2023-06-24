import glob
import numpy as np
import cv2 as cv2
import tensorflow as tf
from keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, Dense, Activation
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

    #Same data augmentation for 0 class.
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

history = CNN.fit(train_data_np, train_answers_np, batch_size=45, epochs=9)

x = [i for i in range(len(history.history["loss"]))]
plt.plot(x, history.history["loss"], label="loss")
plt.plot(x, history.history["accuracy"], label="accuracy")
plt.legend(['loss', 'accuracy'])
plt.xlabel("Epochs")
plt.savefig('history.png')

train_pred = CNN.predict(train_data_np)
pred_train_list = [int(i[0] > 0.5) for i in train_pred]
confusion_matrix = tf.math.confusion_matrix(train_answers_np, pred_train_list)

CNN.summary()
print("Train data params: ")
print(f"Accuracy: %.2f" % tf.get_static_value((confusion_matrix[0][0] + confusion_matrix[1][1]) / np.sum(confusion_matrix)))
print(f"Recall\\Sensitivity: %.2f" % tf.get_static_value(confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])))
print(f"Specificity: %.2f" % tf.get_static_value(confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])))
print("----------------------------")

test_pred = CNN.predict(test_data_np)
pred_test_list = [int(i[0] > 0.5) for i in test_pred]

confusion_matrix = tf.math.confusion_matrix(test_answers_np, pred_test_list)
print("Test data params: ")
print(f"Accuracy: %.2f" % tf.get_static_value((confusion_matrix[0][0] + confusion_matrix[1][1]) / np.sum(confusion_matrix)))
print(f"Recall\\Sensitivity: %.2f" % tf.get_static_value(confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])))
print(f"Specificity: %.2f" % tf.get_static_value(confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])))
print("----------------------------")
