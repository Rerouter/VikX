import cv2
import numpy as np
import os
from PIL import Image, ImageFilter, ImageEnhance
from multiprocessing import Pool, freeze_support

Image.MAX_IMAGE_PIXELS = None  # No limit to image size

# define data directories
data_dir_train = "Z:\\AstroImageAI\\TrainData"
data_dir_validation = "Z:\\AstroImageAI\\Validation"

# minimum size
min_width = 1  # update based on your preferences
min_height = 1

# color mode
color_mode = "rgb"

# function to apply distortions
def apply_distortions(image, distortions):
    # apply gaussian blur
    if 'blur' in distortions:
        image = image.filter(ImageFilter.GaussianBlur(radius=15))  # adjust radius as needed

    # reduce contrast
    if 'contrast' in distortions:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(0.5)  # reduce contrast; adjust as needed

    # convert PIL Image to OpenCV Mat for geometric transformations
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # apply star shape distortion
    if 'star_distortion' in distortions:
        rows, cols, ch = image.shape
        src_points = np.float32([[cols/2, rows/2], [0,0], [cols-1,0]])
        dst_points = np.float32([[cols/2, rows/3], [0,0], [cols-1,0]])
        matrix = cv2.getAffineTransform(src_points, dst_points)
        image = cv2.warpAffine(image, matrix, (cols, rows))

    # apply trailing
    if 'trailing' in distortions:
        rows, cols, ch = image.shape
        src_points = np.float32([[0, 0], [0, rows-1], [cols-1, 0]])
        dst_points = np.float32([[cols*0.1, 0], [cols*0.1, rows-1], [cols-1, 0]])  # 10% shift to right
        matrix = cv2.getAffineTransform(src_points, dst_points)
        image = cv2.warpAffine(image, matrix, (cols, rows))

    # convert OpenCV Mat back to PIL Image for haze and gradient application
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # apply haze
    if 'haze' in distortions:
        haze = Image.new('RGB', image.size, color=(210,210,210))
        image = Image.blend(image, haze, alpha=0.1)  # adjust alpha for desired level of haze

    return image

# function to process single image
def process_image(filename):
    image_path = os.path.join(data_dir_train, filename)
    if os.path.isfile(image_path):
        print(f"Image {filename} already exists in the output directory")
        return
    try:
        im = Image.open(os.path.join(data_dir_validation, filename)).convert('RGB')
        im = apply_distortions(im, distortions)
        print(image_path)
        im.save(image_path)
    except OSError:
        print(f"Couldn't process image {filename}")

# distortions to apply
distortions = ['blur', 'haze']

# protect your main code
if __name__ == "__main__":
    freeze_support()  # optional, necessary only if you want to create a Windows executable using py2exe or similar
    # process images using multiprocessing
    with Pool() as p:
        p.map(process_image, os.listdir(data_dir_validation))
