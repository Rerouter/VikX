import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, PReLU, ELU, Multiply, Dense, Reshape, GlobalAveragePooling2D, Input, Conv2DTranspose, Conv2D, BatchNormalization, Dropout, MaxPooling2D,Concatenate,UpSampling2D,Add
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.ndimage import gaussian_gradient_magnitude
import numpy as np
from scipy.optimize import curve_fit
from tensorflow.signal import fft2d, ifft2d, fftshift
import matplotlib.pyplot as plt
#Hyper
# data directories
batch_size = 16
epochz = 100
learning_rate=0.001
data_directory_path = "Z:\\AstroImageAI\\"
data_dir_train = "Z:\\AstroImageAI\\TrainData"  # make sure to use correct slashes for directory paths
data_dir_validation = "Z:\\AstroImageAI\\Validation"

# minimum size
min_width = 1000
min_height = 625

# for grayscale or RGB images
color_mode = "rgb"  # you can change this to 'grayscale' if you want to handle grayscale images

def sharpness_metric(y_true, y_pred):
    y_true_fft = fftshift(fft2d(tf.cast(y_true, tf.complex64)))
    y_pred_fft = fftshift(fft2d(tf.cast(y_pred, tf.complex64)))

    magnitude_true = tf.abs(y_true_fft)
    magnitude_pred = tf.abs(y_pred_fft)

    sharpness_loss = tf.reduce_mean(tf.square(magnitude_true - magnitude_pred))

    return sharpness_loss

def preprocess_image(img):
    # resize the image if it's smaller than the minimum size
    img = img.resize((min(min_width, img.width), min(min_height, img.height)))

    # convert the image to RGB or grayscale
    img = img.convert(color_mode.upper())

    return np.array(img)
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

# load images
train_images = []
train_shapes = []  # store the original shapes of the images
for filename in os.listdir(data_dir_train):
    im = Image.open(os.path.join(data_dir_train, filename))
    im_shape = im.size  # store the original shape
    im = preprocess_image(im)
    train_images.append(im)
    train_shapes.append(im_shape)

val_images = []
val_shapes = []  # store the original shapes of the images
for filename in os.listdir(data_dir_validation):
    im = Image.open(os.path.join(data_dir_validation, filename))
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
train_images = train_images / 255.0
val_images = val_images / 255.0
#define model
def squeeze_excite_block(input, ratio=16):
    channels = input.shape[-1]
    se = GlobalAveragePooling2D()(input)
    se = Reshape((1, 1, channels))(se)

    # Add a Convolutional layer to capture spatial information
    se = Conv2D(channels // ratio, kernel_size=1, activation='relu')(se)

    # Modify the attention mechanism to focus on nebulae
    se = Conv2D(channels, kernel_size=1, activation='sigmoid')(se)

    return Multiply()([input, se])
def star_attention_block(input, ratio=16):
    channels = input.shape[-1]
    se = GlobalAveragePooling2D()(input)
    se = Reshape((1, 1, channels))(se)

    # Add a Convolutional layer to capture spatial information
    se = Conv2D(channels // ratio, kernel_size=1, activation='relu')(se)

    # Modify the attention mechanism to focus on stars
    se = Conv2D(channels, kernel_size=1, activation='sigmoid')(se)

    return Multiply()([input, se])
datagen = ImageDataGenerator(
    rotation_range=359,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')
# fit parameters from data
datagen.fit(train_images)
num_channels = 1 if color_mode == 'grayscale' else 3
inputs = Input(shape=(None, None, num_channels))
alphaV=0.1
# Encoder
conv1 = Conv2D(64, 3, padding='same')(inputs)
conv1 = LeakyReLU(alpha=alphaV)(conv1)

conv2 = Conv2D(64, 3, padding='same', dilation_rate=2)(conv1)
conv2 = squeeze_excite_block(conv2)
conv2 = LeakyReLU(alpha=alphaV)(conv2)
conv2 = Conv2D(64, 3, padding='same')(conv2)
conv2 = LeakyReLU(alpha=alphaV)(conv2)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

skip1 = conv2

conv3 = Conv2D(128, 3, padding='same')(pool1)
conv3 = squeeze_excite_block(conv3)
conv3 = LeakyReLU(alpha=alphaV)(conv3)
conv3 = Conv2D(128, 3, padding='same')(conv3)
conv3 = LeakyReLU(alpha=alphaV)(conv3)

conv4 = Conv2D(128, 3, padding='same')(conv3)
conv4 = LeakyReLU(alpha=alphaV)(conv4)
conv4 = Conv2D(128, 3, padding='same')(conv4)
conv4 = LeakyReLU(alpha=alphaV)(conv4)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(256, 3, padding='same')(pool2)
conv5 = LeakyReLU(alpha=alphaV)(conv5)
conv5 = Conv2D(256, 3, padding='same')(conv5)
conv5 = LeakyReLU(alpha=alphaV)(conv5)

conv6 = Conv2D(256, 3, padding='same')(conv5)
conv6 = LeakyReLU(alpha=alphaV)(conv6)
conv6 = Conv2D(256, 3, padding='same')(conv6)
conv6 = LeakyReLU(alpha=alphaV)(conv6)

conv7 = Conv2D(256, 3, padding='same', dilation_rate=2)(conv6)
conv7 = LeakyReLU(alpha=alphaV)(conv7)
conv7 = Conv2D(256, 3, padding='same')(conv7)
conv7 = LeakyReLU(alpha=alphaV)(conv7)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)

conv8 = Conv2D(512, 3, padding='same')(pool3)
conv8 = LeakyReLU(alpha=alphaV)(conv8)
conv8 = Conv2D(512, 3, padding='same')(conv8)
conv8 = LeakyReLU(alpha=alphaV)(conv8)

conv9 = Conv2D(512, 3, padding='same')(conv8)
conv9 = LeakyReLU(alpha=alphaV)(conv9)
conv9 = Conv2D(512, 3, padding='same')(conv9)
conv9 = LeakyReLU(alpha=alphaV)(conv9)

conv10 = Conv2D(512, 3, padding='same')(conv9)
conv10 = LeakyReLU(alpha=alphaV)(conv10)
conv10 = Conv2D(512, 3, padding='same')(conv10)
conv10 = LeakyReLU(alpha=alphaV)(conv10)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv10)

# Bottleneck
conv11 = Conv2D(1024, 3, padding='same')(pool4)
conv11 = LeakyReLU(alpha=alphaV)(conv11)
conv11 = Conv2D(1024, 3, padding='same')(conv11)
conv11 = LeakyReLU(alpha=alphaV)(conv11)

# Decoder
upsample1 = UpSampling2D((2, 2))(conv11)
upsample1 = squeeze_excite_block(upsample1)
upsample1 = Conv2D(512, 3, padding='same')(upsample1)
upsample1 = LeakyReLU(alpha=alphaV)(upsample1)
upsample1 = Conv2D(512, 3, padding='same')(upsample1)
upsample1 = LeakyReLU(alpha=alphaV)(upsample1)
upsample1 = tf.image.resize(upsample1, tf.shape(conv10)[1:3])
concat1 = Concatenate()([upsample1, conv10])
conv13 = Conv2D(512, 3, padding='same')(concat1)
conv13 = LeakyReLU(alpha=alphaV)(conv13)
conv13 = Conv2D(512, 3, padding='same')(conv13)
conv13 = LeakyReLU(alpha=alphaV)(conv13)

conv14 = Conv2D(512, 3, padding='same', dilation_rate=2)(conv13)
conv14 = LeakyReLU(alpha=alphaV)(conv14)
conv14 = Conv2D(512, 3, padding='same')(conv14)
conv14 = LeakyReLU(alpha=alphaV)(conv14)

conv15 = Conv2D(512, 3, padding='same')(conv14)
conv15 = LeakyReLU(alpha=alphaV)(conv15)
conv15 = Conv2D(512, 3, padding='same')(conv15)
conv15 = LeakyReLU(alpha=alphaV)(conv15)

upsample2 = UpSampling2D((2, 2))(conv15)
upsample2 = star_attention_block(upsample2)
upsample2 = Conv2D(256, 3, padding='same')(upsample2)
upsample2 = LeakyReLU(alpha=alphaV)(upsample2)
upsample2 = Conv2D(256, 3, padding='same')(upsample2)
upsample2 = LeakyReLU(alpha=alphaV)(upsample2)
upsample2 = tf.image.resize(upsample2, tf.shape(conv7)[1:3])
concat2 = Concatenate()([upsample2, conv7])
conv16 = Conv2D(256, 3, padding='same')(concat2)
conv16 = LeakyReLU(alpha=alphaV)(conv16)
conv16 = Conv2D(256, 3, padding='same')(conv16)
conv16 = LeakyReLU(alpha=alphaV)(conv16)

conv17 = Conv2D(256, 3, padding='same', dilation_rate=2)(conv16)
conv17 = LeakyReLU(alpha=alphaV)(conv17)
conv17 = Conv2D(256, 3, padding='same')(conv17)
conv17 = LeakyReLU(alpha=alphaV)(conv17)

conv18 = Conv2D(256, 3, padding='same')(conv17)
conv18 = LeakyReLU(alpha=alphaV)(conv18)
conv18 = Conv2D(256, 3, padding='same')(conv18)
conv18 = LeakyReLU(alpha=alphaV)(conv18)

upsample3 = UpSampling2D((2, 2))(conv18)
upsample3 = squeeze_excite_block(upsample3)
upsample3 = Conv2D(128, 3, padding='same')(upsample3)
upsample3 = LeakyReLU(alpha=alphaV)(upsample3)
upsample3 = Conv2D(128, 3, padding='same')(upsample3)
upsample3 = LeakyReLU(alpha=alphaV)(upsample3)
upsample3 = tf.image.resize(upsample3, tf.shape(conv4)[1:3])
concat3 = Concatenate()([upsample3, conv4])
conv19 = Conv2D(128, 3, padding='same')(concat3)
conv19 = LeakyReLU(alpha=alphaV)(conv19)
conv19 = Conv2D(128, 3, padding='same')(conv19)
conv19 = LeakyReLU(alpha=alphaV)(conv19)

conv20 = Conv2D(128, 3, padding='same')(conv19)
conv20 = LeakyReLU(alpha=alphaV)(conv20)
conv20 = Conv2D(128, 3, padding='same')(conv20)
conv20 = LeakyReLU(alpha=alphaV)(conv20)

upsample4 = UpSampling2D((2, 2))(conv20)
upsample4 = star_attention_block(upsample4)
upsample4 = Conv2D(64, 3, padding='same')(upsample4)
upsample4 = LeakyReLU(alpha=alphaV)(upsample4)
upsample4 = Conv2D(64, 3, padding='same')(upsample4)
upsample4 = LeakyReLU(alpha=alphaV)(upsample4)
upsample4 = tf.image.resize(upsample4, tf.shape(conv2)[1:3])
concat4 = Concatenate()([upsample4, conv2])
conv21 = Conv2D(64, 3, padding='same')(concat4)
conv21 = LeakyReLU(alpha=alphaV)(conv21)
conv21 = Conv2D(64, 3, padding='same')(conv21)
conv21 = LeakyReLU(alpha=alphaV)(conv21)

conv22 = Conv2D(64, 3, padding='same', dilation_rate=3)(conv21)
conv22 = LeakyReLU(alpha=alphaV)(conv22)
conv22 = Conv2D(64, 3, padding='same')(conv22)
conv22 = LeakyReLU(alpha=alphaV)(conv22)

# Residual connection
residual = Conv2D(3, 1, padding='same')(conv22)
output = Add()([inputs, residual])

# Create the model
model = Model(inputs=inputs, outputs=output)

# print the model summary
model.summary()
now = datetime.now()

# format this into a string
date_time = now.strftime("%Y%m%d_%H%M%S")

# append this to the filename
filename = f"VikX_{date_time}_Batch_{batch_size}_Epoch_{epochz}.h5"


# save the model
model.save(data_directory_path+filename)
# train the model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
history = model.fit(datagen.flow(train_images, train_images, batch_size=batch_size),
                    validation_data=(val_images, val_images),
                    epochs=epochz)
sample_image_path = r"C:\Users\Sidharth Chandra\Desktop\VikX\Validation\ANathanNAstarless.tif"
sample_image = Image.open(sample_image_path)
sample_image = preprocess_image_2(sample_image)
sample_image = sample_image / 255.0
sample_image = np.expand_dims(sample_image, axis=0)
output_image = model.predict(sample_image)
output_image = output_image.squeeze() * 255.0
output_image = Image.fromarray(output_image.astype(np.uint8))
output_image.show()
file_path = f"Z:\\AstroImageAI\\output_image_{date_time}.png"

# Save the image
output_image.save(file_path)
output_image.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

# save the figure to a file
plt.savefig('Z:\\AstroImageAI\\learning_curve.png')


plt.show()
