# CIFAR Classification Task with simple CNN and pretrained model with PyTorch
## Overview
This repository contains PyTorch implementations of **Convolutional Neural Networks (CNNs)** and **ResNet-18** for image classification on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.  
The project compares different models:
- **SimpleCNN** – a small custom convolutional model  
- **CustomCNN** – a deeper custom network  
- **ResNet-18** – pretrained on ImageNet and fine-tuned for CIFAR-10  

## Description
This project continues from the previous project on [FashionMNIST](https://github.com/ignsagita/neuralnets-fashionmnist).
The notebook includes data handling, train/validation/test splits, early stopping, model saving, and an integrated pipeline for the training process.

## Dataset
In Fashion MNIST classification, the subject is always in the center of a 28x28 image. This means the network only needs to get important features from a fixed area.
However, in this section, we will use the CIFAR-10 dataset, which contains 60,000 32x32 color images (RGB) in 10 classes, with 6,000 images per class.

**Labels** <br>
Each training and test example is assigned to one of the following labels:<br>
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

> Note: The dataset is **not included in this repository**. The notebook downloads the dataset automatically from [this website](https://www.cs.toronto.edu/~kriz/cifar.html).

## Result


## Quick Start
- Clone this repository: git clone https://github.com/ignsagita/cnn-resnet-cifar.git cd cnn-resnet-cifar
- Install dependencies pip install -r requirements.txt
- Recommended Setup: For the best experience, **run this notebook on [Google Colab](https://colab.research.google.com/)** 
- In Colab, **enable GPU support** by going to: `Runtime > Change runtime type > Hardware accelerator > GPU`


---
