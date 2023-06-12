import cv2
import numpy as np
import os
from PIL import Image, ImageFilter, ImageEnhance

# define data directories
data_dir_train = "C:\\Users\\Sidharth Chandra\\Desktop\\VikX\\Train"  # update with your directory paths
data_dir_validation = "C:\\Users\\Sidharth Chandra\\Desktop\\VikX\\Validation"

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
        # adjust parameters as needed
        rows, cols, ch = image.shape
        src_points = np.float32([[cols/2, rows/2], [0,0], [cols-1,0]])
        dst_points = np.float32([[cols/2, rows/3], [0,0], [cols-1,0]])
        matrix = cv2.getAffineTransform(src_points, dst_points)
        image = cv2.warpAffine(image, matrix, (cols, rows))

    # apply trailing
    if 'trailing' in distortions:
        # adjust parameters as needed
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

    # apply gradient
    """
    if 'gradient' in distortions:
        gradient = Image.new('L', image.size, color=128)
        image = Image.blend(image, gradient, alpha=0.1)  # adjust alpha for desired level of gradient
    """

    return image

# example usage:
distortions = ['blur', 'haze']

# load and process training images
train_images = []
for filename in os.listdir(data_dir_validation):
    im = Image.open(os.path.join(data_dir_validation, filename)).convert('RGB')
    im = apply_distortions(im, distortions)
    image_path=os.path.join(data_dir_train,filename)
    print(image_path)
    im.save(image_path)
