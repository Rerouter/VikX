import cv2
import numpy as np
import os
import random
from PIL import Image, ImageFilter, ImageEnhance

# define data directories
data_dir_train = "Z:\\AstroImageAI\\TrainData"  # update with your directory paths
data_dir_validation = "Z:\\AstroImageAI\\Validation"

#randomization
def apply_distortions(image, distortions):
    if 'blur' in distortions:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 5)))

    if 'contrast' in distortions:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.5, 1.5))

    if 'rotate' in distortions:
        image = image.rotate(random.randint(-30, 30))

    if 'flip' in distortions:
        if random.random() > 0.5:  # flip with 50% probability
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

    if 'brightness' in distortions:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.7, 1.3))

    # convert PIL Image to OpenCV Mat for geometric transformations
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # rest of your code for star_distortion, trailing, haze, etc...

    return image

# load and process training images
train_images = []
for filename in os.listdir(data_dir_validation):
    available_distortions = ['blur', 'haze', 'contrast', 'rotate', 'flip', 'brightness']
    num_distortions_to_apply = random.randint(1, len(available_distortions))
    distortions = random.sample(available_distortions, num_distortions_to_apply)

    im = Image.open(os.path.join(data_dir_validation, filename)).convert('RGB')
    im = apply_distortions(im, distortions)
    image_path = os.path.join(data_dir_train,filename)
    im.save(image_path)
