Plant Disease Identification System (GAN-Augmented)

Overview
This project addresses the challenge of data scarcity and class imbalance in agricultural computer vision. By implementing a Generative Adversarial Network (DCGAN), the pipeline synthesizes artificial training samples to bolster a MobileNetV2 classifier, enabling the detection of 38 distinct plant disease classes across multiple species.

Key Features
Synthetic Augmentation: Uses a Deep Convolutional GAN to generate realistic leaf images, mitigating over-fitting in rare disease classes.
Transfer Learning: Leverages a pre-trained MobileNetV2 backbone, optimized for mobile-ready, lightweight inference.
Multi-Class Scalability: Trained to recognize 38 different categories of healthy and diseased plant leaves.
Real-time Inference: Includes a processing script that transforms raw $128 \times 128$ RGB images into labeled diagnoses with confidence scores.

Technical Stack
Core: Python, TensorFlow, Keras
Models: DCGAN (Generative), MobileNetV2 (Discriminative/Classification)
Data Science: NumPy, Matplotlib, Scikit-learn
Image Processing: OpenCV, PIL

Model Architecture
The Generator: Utilizes Conv2DTranspose layers with BatchNormalization and LeakyReLU activations to transform random noise into 128 X 128 synthetic leaf images.
The Classifier: Employs MobileNetV2 with a custom top-level architecture:
Global Average Pooling
Dropout (for regularization)
Softmax Output (38 classes)

Dataset
The model is trained on a comprehensive dataset of plant leaf images, covering 38 classes (e.g., Tomato Bacterial Spot, Potato Early Blight, Healthy Apple, etc.). GAN-generated images were used to balance classes that originally had fewer samples.
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
