import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import numpy as np
from tensorflow.keras.models import Model
min_width = 2000  # you can change this to your preferred minimum width
min_height = 2000
color_mode = "rgb"
def preprocess_image(img):
    # resize the image if it's smaller than the minimum size
    img = img.resize((min(min_width, img.width), min(min_height, img.height)))

    # convert the image to RGB or grayscale
    img = img.convert(color_mode.upper())
    return np.array(img)
def re_process_image(img,w,h):
    img=img.resize((w,h))
    return img

new_model=tf.keras.models.load_model("VikX.h5")
new_model.summary()
sample_image_path = r"C:\Users\Sidharth Chandra\Desktop\VikX\Validation\ANathanNAstarless.tif"
sample_image = Image.open(sample_image_path)
width=sample_image.width
height=sample_image.height
display=sample_image.resize((2000,2000))
display.show()
sample_image = preprocess_image(sample_image)
print(sample_image)
sample_image = sample_image / 255.0
sample_image = np.expand_dims(sample_image, axis=0)
output_image = new_model.predict(sample_image)
output_image = output_image.squeeze() * 255.0
output_image = Image.fromarray(output_image.astype(np.uint8))
output_image.show()
