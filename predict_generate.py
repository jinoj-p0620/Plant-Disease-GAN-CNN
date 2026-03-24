import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
# Load the saved model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Load class names
with open('classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_disease(img_path):
    # Load and pre-process image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    result = class_names[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)
    return result, confidence

if __name__ == "__main__":
    path = r'image.jpg'
    disease, conf = predict_disease(path)
    print(f"\nPrediction: {disease}")
    print(f"Confidence: {conf:.2f}%")