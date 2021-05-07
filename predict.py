import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from model_utils import process_image, predict

parser = argparse.ArgumentParser(description='Image Classifier')
parser.add_argument('image_path', action="store")
parser.add_argument('model', action="store")
parser.add_argument('--top_k', action="store", default=1, type=int, 
                    help='Return the top k most likely classes')
parser.add_argument('--category_names', action="store",
                    help='Use a label_map.json file to map labels to flower names')

results = parser.parse_args()

# setting arguments
image_path = results.image_path
keras_model_path = results.model
k = results.top_k
class_names = results.category_names

# loading model
keras_model = tf.keras.models.load_model(keras_model_path, 
                                         custom_objects = {'KerasLayer': hub.KerasLayer}, 
                                         compile=False)
#keras_model.summary()

# calculating probabilities
probs, classes, names = predict(image_path, keras_model, k)

# printing results
print("\n########## Prediction ##########")
print(f"Probabilities: {np.round(probs,3)}")
print(f"Classes: {classes}")

if class_names == "label_map.json":
    print(f"Class names: {names}")
elif class_names == None:
    print("No category names file provided")
else:
    print("Error: Invalid category names file")


    
#pip install -q -U "tensorflow-gpu==2.0.0b1"
#pip install -q -U tensorflow_hub

####### Basic usage:

# $ python predict.py ./test_images/cautleya_spicata.jpg my_model.h5

####### Options:

# Return the top 3 most likely classes:
# $ python predict.py ./test_images/cautleya_spicata.jpg my_model.h5 --top_k 3

# Use a label_map.json file to map labels to flower names:
# $ python predict.py ./test_images/cautleya_spicata.jpg my_model.h5 --category_names label_map.json
