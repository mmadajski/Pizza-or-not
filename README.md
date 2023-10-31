# Pizza Classifier with Customizable Convolutional and Residual Neural Networks

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Experiment Tracking](#experiment-tracking)
---

## Introduction

This project is a pizza image classifier that utilizes Convolutional and Residual Neural Networks (ResNets) to classify whether an image contains pizza or not. The key features of this project are:

- **Customizable Neural Networks:** You can tailor the architecture and hyperparameters of the neural networks to fit your specific use case. All network configurations are specified in a YAML file, making it easy to experiment with different setups.

- **Experiment Tracking:** We use the MLflow package to manage and track experiments, making it simple to compare and analyze different model configurations and their performance.

- **Binary Classification:** The primary task of this project is binary classification: determining whether an image contains pizza or not. It's perfect for any application that requires pizza detection!

---

## Data

The dataset contains 1966 images of various resolutions. 
Source: [kaggle](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza)


---

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine:
```bash
git clone https://github.com/mmadajski/Pizza-or-not
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Download the data from Kaggle platform: [kaggle](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza).
4. Unzip data inside the project.
5. Customize the network parameters by editing the `config.yaml` file to suit your specific requirements. This file contains configurations for the network architecture, training parameters, and more.
6. Train the models by running Pizza.py.
7. You can monitor and manage your experiments using MLflow.
8. Visualize the results and choose the best model configuration for your needs.

---

## Project Structure

The project structure is as follows:

Pizza-or-not/  
├── mlruns/  
├── Pizza.py  
├── Utils.py  
├── config.yaml  
├── requirements.txt
└── README.md  

- `mlruns/`: Directory where experiment logs and artifacts are stored.
- `config.yaml`: YAML file for network customization.
- `Pizza.py`: Script to train your model using the specified configuration.
- `Utils.py`: Helper functions for the main script.
- `requirements.txt`: Project requirements.
- `README.md`: Project description.

---

## Customization

You can customize your network by editing the `config.yaml` file. This YAML file allows you to specify various parameters, including:

- Network architecture (both the Convolutional and ResNet).
- Hyperparameters (e.g., batch size, epochs).
- Data augmentation settings.
- Run name and description.

By modifying this configuration file, you can easily experiment with different network configurations and training options.

---

## Experiment Tracking

I use the MLflow package for experiment tracking. With MLflow, you can easily manage and monitor your experiments, track metrics, and log artifacts. To launch the MLflow UI for monitoring your experiments, use the following command:
```bash
mlflow ui
```
You can access the UI by opening a web browser and navigating to the provided URL.

---


Feel free to reach out if you have any questions or suggestions!

Happy pizza classification! 🍕🔍
