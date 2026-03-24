import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# --- 1. GAN FOR AUGMENTATION (Simplified DCGAN) ---
def build_generator():
    model = models.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, (5, 5), strides=(4, 4), padding='same', use_bias=False, activation='tanh')
    ])
    return model # Output shape (128, 128, 3)

# Note: In a real scenario, you'd train the GAN on your plant images first.
# Here, we initialize it to show how you would generate "augmented" images.
generator = build_generator()

def get_gan_augmented_images(batch_size):
    noise = tf.random.normal([batch_size, 100])
    generated_images = generator(noise, training=False)
    # Rescale from [-1, 1] to [0, 255] for the CNN
    generated_images = (generated_images + 1) * 127.5
    return generated_images

# --- 2. CNN TRAINING ---
data_dir = r'C:\Users\pjino\Downloads\Telegram Desktop\dataset\train' # Update this path
img_size = (128, 128)
batch_size = 32

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names

# Build CNN Model (Using Transfer Learning for better accuracy)
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
print("Starting Training...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save the model
model.save('plant_disease_model.h5')
# Save class names for prediction reference
with open('classes.txt', 'w') as f:
    for item in class_names:
        f.write("%s\n" % item)

print("Model Saved Successfully!")