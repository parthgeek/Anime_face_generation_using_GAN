# Anime Face Generation using Generative Adversarial Network (GAN)

This project demonstrates the use of Generative Adversarial Networks (GANs) to generate images of anime characters. GANs consist of two networks: a **Generator** that creates fake images from random noise, and a **Discriminator** that differentiates between real and fake images. This adversarial process helps the generator learn to produce increasingly realistic images.

In this project, we use the [Anime Face Dataset](https://www.kaggle.com/splcher/animefacedataset) to train a GAN to generate 64x64 pixel images of anime faces.

## Table of Contents
1. [Introduction](#introduction)
2. [Main Components](#main-components)
3. [Project Setup](#project-setup)
4. [Download the Dataset](#dataset)
5. [Data Preprocessing](#Data-Preprocessing)
6. [Model Architecture](#model-architecture)
7. [Training](#training)
8. [Results](#results)
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)

---

## Introduction

Generative Adversarial Networks (GANs) are a class of neural networks used for unsupervised learning tasks, particularly **generative modeling**, where the goal is to generate new samples that resemble a training dataset. In this project, we implement a GAN to generate anime faces using the **Anime Face Dataset**. The architecture consists of a **Generator** and a **Discriminator**, where the generator tries to fool the discriminator by generating fake anime faces, while the discriminator tries to distinguish between real and fake images.

### Overview of GANs

- **Generator**: Takes random noise as input and generates an image.
- **Discriminator**: Takes an image (real or fake) and predicts if it's real (from the dataset) or fake (from the generator).

The networks are trained in tandem, where the Discriminator becomes better at identifying real vs. fake images, while the Generator improves at producing images that can fool the Discriminator.

## Main Components:
1. Data Loading and Preprocessing: Loads the Anime Face Dataset from Kaggle, resizes images, and normalizes pixel values.
2. Model Architecture: Defines the Generator and Discriminator architectures using Convolutional Neural Networks (CNNs).
3. Training Loop: The networks are trained in an adversarial manner to compete against each other, improving their respective tasks.

## Project Setup

### Prerequisites

You need to have Python 3.x installed. The project requires several Python libraries:

- `torch` (PyTorch)
- `torchvision`
- `matplotlib`
- `opendatasets`
- `jovian`
  
Setup
Install Dependencies: You can install the required libraries using the following commands:

For Linux/Windows (No GPU):
```bash
pip install numpy matplotlib torch torchvision torchaudio
```
For Linux/Windows (GPU):
```bash
pip install numpy matplotlib torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
## Download the Dataset:
```python
import opendatasets as od
dataset_url = 'https://www.kaggle.com/splcher/animefacedataset'
od.download(dataset_url)
```
## Data Preprocessing
The images are loaded using torchvision.datasets.ImageFolder and resized to 64x64 pixels. The pixel values are normalized to be in the range (-1, 1) for better training performance.


## Model Architectures
1. Discriminator:
The Discriminator is a CNN that classifies images as real or fake.
It uses LeakyReLU activation functions and a sigmoid output layer for binary classification.
```python

discriminator = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    # ... (additional layers)
    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    nn.Flatten(),
    nn.Sigmoid()
)
```
2. Generator:
The Generator is another CNN that starts with a random noise vector and outputs a 64x64 image.
It uses ReLU activation in all layers except the output layer, which uses a tanh function.
```python
generator = nn.Sequential(
    nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    # ... (additional layers)
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
)
```

## Training
The GAN is trained using the standard binary cross-entropy loss function. We alternate between training the Discriminator and the Generator:

Discriminator: Maximizes the probability of correctly classifying real and fake images.
Generator: Trains to produce images that can fool the Discriminator.

## Results
By the end of the training, the Generator will have learned to create anime faces that are visually indistinguishable from real faces in the dataset.
### Sample Generated Images (1 to 25 epochs):
After 1 epoch:
![1epoch](https://github.com/user-attachments/assets/3a7fe4ff-d5ea-48ed-89e3-c3834e330043)
After 5 epoch:
![5epoch](https://github.com/user-attachments/assets/7001d01c-417b-4593-8f6a-6fc0a9e0e7b5)
After 15 epoch:
![10epoch](https://github.com/user-attachments/assets/7f1d8772-56ea-4cc1-9e99-65a809b223a5)
After 20 epoch:
![20epoch](https://github.com/user-attachments/assets/ee5d0bf8-0239-44d3-bbde-4f17b080b584)
After 25 epoch:
![25epoch](https://github.com/user-attachments/assets/bab5a349-757e-460b-8d3c-95446b6f3303)

Discriminator and Generator Loss During Training
![Loss During Training](https://github.com/user-attachments/assets/cc9b8a56-3e9e-4a2c-b384-d1307eb3df98)
Real and Fake Score of the Discriminator
![Score of the Discriminator](https://github.com/user-attachments/assets/18e0488d-228a-4b32-9004-2e67852b53a8)

