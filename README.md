# CIFAR-10 Image Classification using CNNs

This project implements a **Convolutional Neural Network (CNN)** to classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), a standard benchmark dataset in computer vision. The model is trained, evaluated, and improved using different techniques in this educational deep learning assignment.

---

## Objective

- Understand and apply CNNs for image classification.
- Explore, preprocess, and visualize the CIFAR-10 dataset.
- Build a basic CNN using TensorFlow/Keras.
- Train and evaluate the model with accuracy/loss visualization.
- Experiment with different optimizers to improve performance.

---

## Dataset: CIFAR-10

- **60,000** color images (32x32 pixels, RGB)
- **10 Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Split: **50,000** training images | **10,000** test images

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install all dependencies using:

```bash
pip install tensorflow matplotlib seaborn scikit-learn
```

---

## Tasks Overview

### Data Exploration and Preparation
- Loaded the CIFAR-10 dataset using Keras.
- Displayed sample images with their labels.
- Normalized image data to the range **[0, 1]**.
- Printed shape and label distribution.

### Build and Train CNN Model
- Built a CNN using:
  - `Conv2D`, `ReLU`, `MaxPooling2D`, `Dropout`, and `Dense` layers.
- Trained for **15 epochs** using the **Adam** optimizer.
- Plotted training and validation **accuracy and loss curves**.

### Evaluate the Model
- Evaluated test set accuracy.
- Generated a **confusion matrix** and **classification report**.
- Displayed examples of correct and incorrect predictions.

###  Experimentation
- Re-trained the CNN using the **SGD** optimizer.
- Compared performance between **Adam** and **SGD**.

---

## Results

| Optimizer | Test Accuracy |
|-----------|---------------|
| Adam      | ~0.7062       |
| SGD       | ~0.5292       |

---

## Key Learnings

- How to build and train CNNs using Keras.
- Importance of regularization techniques like **Dropout**.
- Significant impact of optimizer choice (**Adam** vs. **SGD**).
- Visualizations help detect **overfitting** or **underfitting**.
