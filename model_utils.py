import numpy as np
import tensorflow as tf
import json
from PIL import Image

with open('label_map.json', 'r') as f:
    class_names = json.load(f)
    

def process_image(raw_image):
    tf_image = tf.convert_to_tensor(raw_image)
    tf_image = tf.image.resize(tf_image, (224, 224))
    tf_image /= 255
    return tf_image.numpy()


def predict(image_path, model, top_k):
    raw_image = Image.open(image_path)
    raw_image = np.asarray(raw_image)
    proc_image = process_image(raw_image)
    proc_image = np.expand_dims(proc_image, axis=0)
    preds = model.predict(proc_image)
    probs, classes = tf.math.top_k(preds, k = top_k)
    classes = classes[0] + 1
    names = [class_names[str(x)] for x in classes.numpy()]
    
    return probs[0].numpy(), classes.numpy(), names


