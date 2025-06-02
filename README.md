# CIFAR-10 Image Classification using CNNs

This project implements a Convolutional Neural Network (CNN) to classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), a standard benchmark dataset in computer vision. The model is trained, evaluated, and improved using different techniques in this educational deep learning assignment.

---

## Objective

- Understand and apply CNNs for image classification.
- Explore, preprocess, and visualize the CIFAR-10 dataset.
- Build a basic CNN using TensorFlow/Keras.
- Train and evaluate the model with visualization.
- Experiment with optimizer changes for performance improvement.

---

## Dataset: CIFAR-10

- 60,000 images (32x32 pixels, RGB)
- 10 Classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
- Training: 50,000 images | Testing: 10,000 images

---

## How to Run

### Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

### Installation

`pip install tensorflow matplotlib seaborn scikit-learn`

---

## Data Exploration and Preparation
- Loaded CIFAR-10 dataset
- Displayed sample images with labels
- Normalized image data to [0, 1]
- Displayed label distribution

## Build and Train CNN Model
- Built a CNN using Conv2D, MaxPooling, Dropout, Dense layers
- Trained model for 15 epochs with Adam optimizer
- Plotted accuracy/loss over epochs

## Evaluate the Model
- Evaluated test accuracy
- Generated confusion matrix and classification report
- Displayed correct and incorrect predictions

## Experimentation
- Re-trained model using SGD optimizer
- Compared performance between Adam and SGD



