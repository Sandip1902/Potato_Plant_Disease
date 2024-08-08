import joblib
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from io import BytesIO
#data = "1ed357f9-f036-4bf2-b180-1588976eb116___RS_LB 3005.JPG"

def model_potato(data):
    jpeg_image = Image.open(data)

    # Use a different name for the saved image to avoid conflicts
    saved_image_path = "saved_image.jpg"
    jpeg_image.save(saved_image_path)

    with open(saved_image_path, 'rb') as file:
        image_data = file.read() 

    def read_file_as_image(img) -> np.ndarray:
        image = np.array(Image.open(BytesIO(img)))
        return image

    CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
    image = read_file_as_image(image_data)

    img_batch = np.expand_dims(image, 0)

    classifier = joblib.load("potato_disease_classification.pkl")

    predictions = classifier.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

    #confidence = np.max(predictions[0])
    return predicted_class


#print(model_potato(data))
