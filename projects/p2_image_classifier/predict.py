import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image
import argparse

# Creat a parser
parser = argparse.ArgumentParser(description="Classify Flowers Script")

# Positional arguments 
parser.add_argument("image_path")
parser.add_argument("saved_keras_model")
parser.add_argument("-k", "--top_k", default=3, type=int)
parser.add_argument("-n", "--category_names", default="./label_map.json",type=str)


args = parser.parse_args()   
image_path = args.image_path
saved_keras_model = args.saved_keras_model
top_k = args.top_k
category_names = args.category_names

  


# Load a JSON file that maps the class values to other category names
with open(category_names, 'r') as f:
    class_names = json.load(f)


# Load the saved Keras model 
loaded_keras_model = tf.keras.models.load_model(saved_keras_model, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
print(loaded_keras_model.summary())
image_size = 224

# Image resize and normalization
def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()


# Predict the top K flower classes along with associated probabilities
def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)
    model_prediction = model.predict(expanded_image)
    top_k_probs, top_k_classes = tf.nn.top_k(model_prediction, k=top_k)
    top_k_probs = list(top_k_probs.numpy()[0])
    top_k_classes = list(top_k_classes.numpy()[0])  
    return top_k_probs, top_k_classes


# Call the function that retures the top K classes along with associated probabilities
top_k_probs, top_k_classes = predict(image_path, loaded_keras_model, top_k) 
print('flower labels along with probabilities:', top_k_classes, top_k_probs)

for flower in range(len(top_k_classes)): 
    print('Name:', class_names.get(str(top_k_classes[flower]+1)))
    print('Probabilty:', top_k_probs[flower])