import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, PReLU, ELU, Multiply, Dense, Reshape, GlobalAveragePooling2D, Input, Conv2DTranspose, Conv2D, BatchNormalization, Dropout, \
    MaxPooling2D, Concatenate, UpSampling2D, Add
from PIL import Image
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.ndimage import gaussian_gradient_magnitude
import numpy as np
from scipy.optimize import curve_fit
from tensorflow.signal import fft2d, ifft2d, fftshift

# Hyper
batch_size = 2
learning_rate = 0.001
# data directories
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_train = os.path.join(base_dir, 'Train')
data_dir_validation = os.path.join(base_dir, 'Validation')

# minimum size
min_width = 256  # you can change this to your preferred minimum width
min_height = 256  # you can change this to your preferred minimum height

# for grayscale or RGB images
color_mode = "rgb"  # you can change this to 'grayscale' if you want to handle grayscale images


def sharpness_metric(y_true, y_pred):
    y_true_fft = fftshift(fft2d(tf.cast(y_true, tf.complex64)))
    y_pred_fft = fftshift(fft2d(tf.cast(y_pred, tf.complex64)))

    magnitude_true = tf.abs(y_true_fft)
    magnitude_pred = tf.abs(y_pred_fft)

    sharpness_loss = tf.reduce_mean(tf.square(magnitude_true - magnitude_pred))

    return sharpness_loss


def preprocess_image(img, target_size=(min_width, min_height), color_mode='rgb'):
    height, width, _ = img.shape

    # find the larger dimension
    max_dim = max(height, width)

    # find scale factor
    scale_factor = target_size[0] / max_dim

    # calculate new dimensions
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)

    # resize the image so the larger dimension fits the target size
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # pad the image to fit the target size
    delta_w = target_size[1] - new_width
    delta_h = target_size[0] - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # convert the image to RGB or grayscale
    if color_mode == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_mode == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def preprocess_image_2(img):
    # resize the image if it's smaller than the minimum size
    img = img.resize((min(2000, img.width), min(2000, img.height)))

    # convert the image to RGB or grayscale
    img = img.convert(color_mode.upper())

    return np.array(img)


def pad_images(images):
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    padded_images = []
    for image in images:
        pad_height = max_height - image.shape[0]
        pad_width = max_width - image.shape[1]
        padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        padded_images.append(padded_image)
    return np.array(padded_images)


def equalize_and_normalize_hist(img):
    # Compute the maximum pixel value based on the data type of the image
    max_pixel_value = np.iinfo(img.dtype).max

    # Reshape the image to a 1D array.
    img_flat = img.flatten()

    # Get the histogram of the image.
    hist, bins = np.histogram(img_flat, max_pixel_value+1, [0,max_pixel_value+1])

    # Compute the cumulative distribution function of the histogram.
    cdf = hist.cumsum()

    # Mask all zeros in the cumulative distribution function.
    cdf_m = np.ma.masked_equal(cdf, 0)

    # Perform the histogram equalization.
    cdf_m = (cdf_m - cdf_m.min()) * max_pixel_value / (cdf_m.max() - cdf_m.min())

    # Fill masked pixels with zero and cast the result to the appropriate integer datatype.
    cdf = np.ma.filled(cdf_m, 0).astype(img.dtype)

    # Use the cumulative distribution function as a lookup table to equalize the pixels in the image.
    img_eq_flat = cdf[img_flat]

    # Reshape the equalized array back into the original 3D shape.
    img_eq = img_eq_flat.reshape(img.shape)

    # Normalize the equalized image to the range [0,1].
    img_norm = img_eq / max_pixel_value

    return img_norm


def open_image_as_np_array(image_path):
    # cv2.IMREAD_UNCHANGED ensures that the image's bit depth is preserved
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # cv2 reads images in BGR format, convert it to RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# load images
train_images = []
train_shapes = []  # store the original shapes of the images
for filename in os.listdir(data_dir_train):
    im = open_image_as_np_array(os.path.join(data_dir_train, filename))
    im_shape = im.size  # store the original shape
    im = preprocess_image(im)
    train_images.append(im)
    train_shapes.append(im_shape)

val_images = []
val_shapes = []  # store the original shapes of the images
for filename in os.listdir(data_dir_validation):
    im = open_image_as_np_array(os.path.join(data_dir_validation, filename))
    im_shape = im.size  # store the original shape
    im = preprocess_image(im)
    val_images.append(im)
    val_shapes.append(im_shape)

# convert lists to numpy arrays
train_images = np.array(train_images)
val_images = np.array(val_images)

# pad images
train_images = pad_images(train_images)
val_images = pad_images(val_images)

# normalize images
train_images = np.array([equalize_and_normalize_hist(img) for img in train_images])
val_images = np.array([equalize_and_normalize_hist(img) for img in val_images])


# define model
def squeeze_excite_block(input, ratio=16):
    channels = input.shape[-1]
    se = GlobalAveragePooling2D()(input)
    se = Reshape((1, 1, channels))(se)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    return Multiply()([input, se])


num_channels = 1 if color_mode == 'grayscale' else 3
inputs = Input(shape=(None, None, num_channels))
alphaV = 0.1
# Encoder
conv1 = Conv2D(64, 3, padding='same')(inputs)
conv1 = LeakyReLU(alpha=alphaV)(conv1)

conv2 = Conv2D(64, 3, padding='same', dilation_rate=2)(conv1)
conv2 = squeeze_excite_block(conv2)
conv2 = LeakyReLU(alpha=alphaV)(conv2)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

skip1 = conv2

conv3 = Conv2D(128, 3, padding='same')(pool1)
conv3 = LeakyReLU(alpha=alphaV)(conv3)

conv4 = Conv2D(128, 3, padding='same')(conv3)
conv4 = LeakyReLU(alpha=alphaV)(conv4)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(256, 3, padding='same')(pool2)
conv5 = LeakyReLU(alpha=alphaV)(conv5)

conv6 = Conv2D(256, 3, padding='same')(conv5)
conv6 = LeakyReLU(alpha=alphaV)(conv6)

conv7 = Conv2D(256, 3, padding='same', dilation_rate=2)(conv6)
conv7 = LeakyReLU(alpha=alphaV)(conv7)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)

conv8 = Conv2D(512, 3, padding='same')(pool3)
conv8 = LeakyReLU(alpha=alphaV)(conv8)

conv9 = Conv2D(512, 3, padding='same')(conv8)
conv9 = LeakyReLU(alpha=alphaV)(conv9)

conv10 = Conv2D(512, 3, padding='same')(conv9)
conv10 = LeakyReLU(alpha=alphaV)(conv10)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv10)

# Bottleneck
conv11 = Conv2D(1024, 3, padding='same')(pool4)
conv11 = LeakyReLU(alpha=alphaV)(conv11)

conv12 = Conv2D(1024, 3, padding='same')(conv11)
conv12 = LeakyReLU(alpha=alphaV)(conv12)

# Decoder
upsample1 = UpSampling2D((2, 2))(conv12)
concat1 = Concatenate()([upsample1, conv10])
conv13 = Conv2D(512, 3, padding='same')(concat1)
conv13 = LeakyReLU(alpha=alphaV)(conv13)

conv14 = Conv2D(512, 3, padding='same', dilation_rate=2)(conv13)
conv14 = LeakyReLU(alpha=alphaV)(conv14)

conv15 = Conv2D(512, 3, padding='same')(conv14)
conv15 = LeakyReLU(alpha=alphaV)(conv15)

upsample2 = UpSampling2D((2, 2))(conv15)
concat2 = Concatenate()([upsample2, conv7])
conv16 = Conv2D(256, 3, padding='same')(concat2)
conv16 = LeakyReLU(alpha=alphaV)(conv16)

conv17 = Conv2D(256, 3, padding='same', dilation_rate=2)(conv16)
conv17 = LeakyReLU(alpha=alphaV)(conv17)

conv18 = Conv2D(256, 3, padding='same')(conv17)
conv18 = LeakyReLU(alpha=alphaV)(conv18)

upsample3 = UpSampling2D((2, 2))(conv18)
concat3 = Concatenate()([upsample3, conv4])
conv19 = Conv2D(128, 3, padding='same')(concat3)
conv19 = LeakyReLU(alpha=alphaV)(conv19)

conv20 = Conv2D(128, 3, padding='same')(conv19)
conv20 = LeakyReLU(alpha=alphaV)(conv20)

upsample4 = UpSampling2D((2, 2))(conv20)
concat4 = Concatenate()([upsample4, conv2])
conv21 = Conv2D(64, 3, padding='same')(concat4)
conv21 = LeakyReLU(alpha=alphaV)(conv21)

conv22 = Conv2D(64, 3, padding='same', dilation_rate=3)(conv21)
conv22 = LeakyReLU(alpha=alphaV)(conv22)

# Residual connection
residual = Conv2D(3, 1, padding='same')(conv22)
output = Add()([inputs, residual])

# Create the model
model = Model(inputs=inputs, outputs=output)

# print the model summary
model.summary()
model.save('VikX.h5')
# train the model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
model.fit(train_images, train_images, validation_data=(val_images, val_images), epochs=300, batch_size=batch_size)
sample_image_path = os.path.join(data_dir_validation, "ANathanNAstarless.tif")
sample_image = Image.open(sample_image_path)
sample_image = preprocess_image_2(sample_image)
sample_image = normalize_image(equalize_hist_color(sample_image))
sample_image = np.expand_dims(sample_image, axis=0)
output_image = model.predict(sample_image)
output_image = output_image.squeeze() * 255.0
output_image = Image.fromarray(output_image.astype(np.uint8))
output_image.show()
