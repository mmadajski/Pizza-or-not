import glob
import keras
import mlflow
import numpy as np
import cv2 as cv2
import tensorflow as tf
from keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, Dense, Activation, BatchNormalization, ReLU, \
    AveragePooling2D
from random import seed
from random import choices
import matplotlib
import yaml
from yaml.loader import SafeLoader
from Utils import build_cnn_network, build_resnet_network, calculate_metrics

mlflow.start_run()

paths_pizza = glob.glob(".//pizza_not_pizza//pizza//*.jpg")[:5]
paths_not_pizza = glob.glob(".//pizza_not_pizza//not_pizza//*.jpg")[:5]

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

train_data_np = np.array(data_train_list) / 255
train_answers_np = np.array(data_train_answer)

test_data_np = np.array(data_test) / 255
test_answers_np = np.array(data_test_answers)

with open('params.yaml') as f:
    parameters = yaml.load(f, Loader=SafeLoader)
    cnn_params = parameters["CNN"]
    resnet_params = parameters["ResNet"]

mlflow.log_param("CNN", cnn_params)
mlflow.log_param("RenNet", resnet_params)

# CNN network
CNN = build_cnn_network(cnn_params)
CNN.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy', 'Recall'])

# ResNet network
ResNet = build_resnet_network(resnet_params)
ResNet.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy', 'Recall'])

history_cnn = CNN.fit(train_data_np, train_answers_np, batch_size=cnn_params["batch_size"], epochs=cnn_params["epochs"])
history_res = ResNet.fit(train_data_np, train_answers_np, batch_size=resnet_params["batch_size"], epochs=resnet_params["epochs"])

for acc, loss in zip(history_cnn.history["accuracy"], history_cnn.history["loss"]):
    mlflow.log_metric("CNN history accuracy", acc)
    mlflow.log_metric("CNN history loss", loss)

for acc, loss in zip(history_res.history["accuracy"], history_res.history["loss"]):
    mlflow.log_metric("Resnet history accuracy", acc)
    mlflow.log_metric("Resnet history loss", loss)

train_pred_cnn = CNN.predict(train_data_np)
pred_train_cnn_list = [int(i[0] > 0.5) for i in train_pred_cnn]
cnn_train_metrics = calculate_metrics(train_answers_np, pred_train_cnn_list)
mlflow.log_metric("CNN train accuracy", cnn_train_metrics[0])
mlflow.log_metric("CNN train recall", cnn_train_metrics[1])
mlflow.log_metric("CNN train specificity", cnn_train_metrics[2])

test_pred_cnn = CNN.predict(test_data_np)
pred_test_cnn_list = [int(i[0] > 0.5) for i in test_pred_cnn]
cnn_test_metrics = calculate_metrics(test_answers_np, pred_test_cnn_list)
mlflow.log_metric("CNN test accuracy", cnn_test_metrics[0])
mlflow.log_metric("CNN test recall", cnn_test_metrics[1])
mlflow.log_metric("CNN test specificity", cnn_test_metrics[2])

train_pred_res = ResNet.predict(train_data_np)
pred_train_res_list = [int(i[0] > 0.5) for i in train_pred_res]
res_net_train_metrics = calculate_metrics(train_answers_np, pred_train_res_list)
mlflow.log_metric("ResNet train accuracy", res_net_train_metrics[0])
mlflow.log_metric("ResNet train recall", res_net_train_metrics[1])
mlflow.log_metric("ResNet train specificity", res_net_train_metrics[2])

test_pred_res = ResNet.predict(test_data_np)
pred_test_res_list = [int(i[0] > 0.5) for i in test_pred_res]
res_net_test_metrics = calculate_metrics(test_answers_np, pred_test_res_list)
mlflow.log_metric("ResNet test accuracy", res_net_test_metrics[0])
mlflow.log_metric("ResNet test recall", res_net_test_metrics[1])
mlflow.log_metric("ResNet test specificity", res_net_test_metrics[2])

mlflow.end_run()