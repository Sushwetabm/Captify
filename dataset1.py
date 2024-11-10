import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import kaggle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import TextVectorization
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter
import re


# Paths for images and captions
IMAGES_PATH = "datasets/flickr8k_images/Images"  # Specify a local path
CAPTIONS_PATH = "datasets/flickr8k_images/captions.txt"  # Your caption file should be here

PREPROCESSED_IMAGES_PATH = "/kaggle/working/preprocessed_images"

# Dataset identifier from Kaggle (for flickr8k dataset)
dataset_identifier = 'adityajn105/flickr8k'

# Download the dataset
kaggle.api.dataset_download_files(dataset_identifier, path=IMAGES_PATH, unzip=True)
print(f"Dataset downloaded to {IMAGES_PATH}")

# Desired image dimensions
IMAGE_SIZE = (299, 299)

# Fixed length allowed for any sequence
SEQ_LENGTH = 25

# Vocabulary size
VOCAB_SIZE = 10000

# Create the folder for preprocessed images if it doesn't exist
os.makedirs(PREPROCESSED_IMAGES_PATH, exist_ok=True)

# Load captions data
def load_captions_data(filename):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()[1:]
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            img_name, caption = line.split(",", 1)
            img_name = os.path.join(IMAGES_PATH, img_name.strip())
            tokens = caption.strip().split()
            if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
                images_to_skip.add(img_name)
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)
                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]

        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]

        return caption_mapping, text_data

# Clean and filter text data
def clean_and_filter_text(captions, min_freq=5):
    token_freq = Counter(" ".join(captions).split())
    cleaned_captions = []
    
    for caption in captions:
        cleaned_caption = []
        for word in caption.split():
            if token_freq[word] >= min_freq:
                cleaned_caption.append(word)
            else:
                cleaned_caption.append("<UNK>")
        cleaned_captions.append(" ".join(cleaned_caption))
    return cleaned_captions

# Custom standardization function
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    strip_chars = "!\"#$%&'()*+,-./:;=?@[\]^_{|}~1234567890"
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# Vectorizer for text data
vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization)

# Load and adapt the vectorizer to filtered text data
captions_mapping, raw_text_data = load_captions_data(CAPTIONS_PATH)
filtered_text_data = clean_and_filter_text(raw_text_data)
vectorization.adapt(filtered_text_data)

# Train-validation-test split
def train_val_split(caption_data, validation_size=0.2, test_size=0.05, shuffle=True):
    all_images = list(caption_data.keys())
    if shuffle:
        np.random.shuffle(all_images)

    train_keys, validation_keys = train_test_split(all_images, test_size=validation_size, random_state=42)
    validation_keys, test_keys = train_test_split(validation_keys, test_size=test_size, random_state=42)

    training_data = {img_name: caption_data[img_name] for img_name in train_keys}
    validation_data = {img_name: caption_data[img_name] for img_name in validation_keys}
    test_data = {img_name: caption_data[img_name] for img_name in test_keys}

    return training_data, validation_data, test_data

# Splitting dataset
train_data, validation_data, test_data = train_val_split(captions_mapping)
print(f"Total samples: {len(captions_mapping)}")
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(validation_data)}")
print(f"Test samples: {len(test_data)}")

# Load the InceptionV3 model for feature extraction
base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',  # Use 'imagenet' instead of a file path
    input_shape=IMAGE_SIZE + (3,)
)

feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Extract and save preprocessed image features
def preprocess_and_save_images(image_folder, output_folder):
    for img_name in os.listdir(image_folder):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(image_folder, img_name)
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
            img_array = tf.expand_dims(img_array, axis=0)
            features = feature_extractor(img_array)
            feature_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}.npy")
            np.save(feature_path, features.numpy())
            print(f"Saved features for {img_name} at {feature_path}")

# Run the preprocessing function
preprocess_and_save_images(IMAGES_PATH, PREPROCESSED_IMAGES_PATH)

import json

# Save captions to a JSON file
def save_captions(caption_mapping, output_file="captions.json"):
    captions_data = {}
    for img_name, captions in caption_mapping.items():
        # Extract the image name without path and extension
        img_key = os.path.splitext(os.path.basename(img_name))[0]
        captions_data[img_key] = captions

    # Write to JSON file
    with open(output_file, "w") as json_file:
        json.dump(captions_data, json_file, indent=4)
    print(f"Captions saved to {output_file}")

# Run the save captions function
save_captions(captions_mapping)
