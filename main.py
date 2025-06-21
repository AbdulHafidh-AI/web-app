# Author Abdul Hafidh
# Handwriting Digit Image Generator
import streamlit as st
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import random

# Load .keras model
model = tf.keras.models.load_model('model/mnist_model.keras')

# Load MNIST dataset (only once)
@st.cache_data
def load_mnist_samples():
    ds = tfds.load('mnist', split='train', as_supervised=True)
    digit_samples = {i: [] for i in range(10)}
    for image, label in tfds.as_numpy(ds):
        digit_samples[label].append(image)
    return digit_samples

digit_samples = load_mnist_samples()

# App UI
def main():
    st.title("Handwriting Digit Image Generator")
    st.write("This app shows a real handwritten digit image and runs it through the classifier model.")

    digit = st.number_input("Enter a digit (0-9):", min_value=0, max_value=9, value=0)

    if st.button("Generate Image"):
        st.write(f"Showing a sample image for digit: {digit}")

        # Get a random image of the selected digit
        image = random.choice(digit_samples[digit])
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
