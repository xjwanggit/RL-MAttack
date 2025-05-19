import tensorflow as tf
import numpy as np

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image to [0, 1]

    # Custom normalization (if necessary, to match training conditions)
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std  # Normalize image

    return image


def prepare_images_for_prediction(image_paths):
    # Convert image paths to tensors and preprocess
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    image_dataset = image_dataset.map(load_and_preprocess_image)
    image_dataset = image_dataset.batch(64)  # Batch size can be set as needed
    return image_dataset

# Assuming the model and placeholders are set up

def predict_features(image_paths, feature_model):
    tf.keras.backend.set_learning_phase(0)
    image_dataset = prepare_images_for_prediction(image_paths)
    try:
        features = feature_model.predict(image_dataset) # Use predict directly on the numpy array
    except Exception as e:
        print(f"Failed to predict features: {str(e)}")
        features = None
    return features

# def predict_features(image_paths, feature_model, item_input):
#     dataset = prepare_images_for_prediction(image_paths)
#     features = feature_model.predict(dataset)
#     return features