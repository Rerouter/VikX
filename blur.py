import cv2
import numpy as np
import os

# define data directories
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_train = os.path.join(base_dir, 'Train')
data_dir_validation = os.path.join(base_dir, 'Validation')

# minimum size
min_width = 1  # update based on your preferences
min_height = 1

# color mode
color_mode = "rgb"


# function to apply distortions
def apply_distortions(image, distortions):
    # apply gaussian blur
    if 'blur' in distortions:
        image = cv2.GaussianBlur(image, (33, 33), 0)  # adjust size as needed

    # reduce contrast
    if 'contrast' in distortions:
        alpha = 0.5  # contrast control; smaller values lead to less contrast
        image = cv2.convertScaleAbs(image, alpha=alpha)

    # apply star shape distortion
    if 'star_distortion' in distortions:
        rows, cols, ch = image.shape
        src_points = np.float32([[cols / 2, rows / 2], [0, 0], [cols - 1, 0]])
        dst_points = np.float32([[cols / 2, rows / 3], [0, 0], [cols - 1, 0]])
        matrix = cv2.getAffineTransform(src_points, dst_points)
        image = cv2.warpAffine(image, matrix, (cols, rows))

    # apply trailing
    if 'trailing' in distortions:
        rows, cols, ch = image.shape
        src_points = np.float32([[0, 0], [0, rows - 1], [cols - 1, 0]])
        dst_points = np.float32([[cols * 0.1, 0], [cols * 0.1, rows - 1], [cols - 1, 0]])  # 10% shift to right
        matrix = cv2.getAffineTransform(src_points, dst_points)
        image = cv2.warpAffine(image, matrix, (cols, rows))

    # apply haze
    if 'haze' in distortions:
        intensity = 0.1
        haze = np.ones(image.shape, image.dtype) * 210  # this creates a white image
        image = cv2.addWeighted(image, 1 - intensity, haze, intensity, 0)  # blending the images

    return image


# example usage:
distortions = ['blur', 'haze']

# load and process training images
for filename in os.listdir(data_dir_validation):
    im = cv2.imread(os.path.join(data_dir_validation, filename), cv2.IMREAD_COLOR)
    im = apply_distortions(im, distortions)
    image_path = os.path.join(data_dir_train, filename)
    print(image_path)
    cv2.imwrite(image_path, im)
