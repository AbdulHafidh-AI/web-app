# Author Abdul Hafidh
import streamlit as st
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import random
import os

# Suppress TensorFlow log warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/mnist_model.keras')

model = load_model()

# Load a subset of MNIST dataset (filtered by digit)
@st.cache_data
def get_dataset():
    return list(tfds.as_numpy(tfds.load('mnist', split='train', as_supervised=True)))

dataset = get_dataset()

# App UI
def main():
    st.title("Handwriting Digit Image Generator")
    st.write("This app shows a real handwritten digit image and runs it through the classifier model.")

    digit = st.number_input("Enter a digit (0-9):", min_value=0, max_value=9, value=0)

    if st.button("Generate Image"):
        st.write(f"Showing a sample image for digit: {digit}")

        # Randomly select image of that digit
        filtered_images = [img for img, lbl in dataset if lbl == digit]
        image = random.choice(filtered_images)
        st.image(image, caption=f"Sample image of digit {digit}", width=150)

        # Prepare image for prediction
        image_norm = image / 255.0
        image_input = np.expand_dims(image_norm, axis=0)

        # Predict
        prediction = model.predict(image_input)
        st.write(f"Model Prediction: {np.argmax(prediction)}")
        st.success("Image shown and prediction complete!")

if __name__ == "__main__":
    main()
