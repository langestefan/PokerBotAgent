# <ASSIGNMENT: Generate and load your data sets. Motivate your choices in the docstrings and comments. This file
# contains a suggested structure; you are free to define your own structure, adjust function arguments etc. Don't forget
# to write appropriate tests for your functionality.>

import os
import random
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current file marks the root directory
TRAINING_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "training_images")  # Directory for storing training images
TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "test_images")  # Directory for storing test images
LABELS = ['J', 'Q', 'K', 'A']  # Possible card labels
IMAGE_SIZE = 32 
ROTATE_MAX_ANGLE = 15
NOISE = np.arange(0,1,0.1) # Generate different noise level for data set

FONTS = [
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'normal', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'italic', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'normal', weight = 'medium')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'normal', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'italic', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'normal', weight = 'medium')),
]  # True type system fonts


def extract_features(img: Image):
    """
    Convert an image to features that serve as input to the image classifier.

    Arguments
    ---------
    img : Image
        Image to convert to features.
    data_dir : str
        Directory in which the images are stored
    Returns
    -------
    features : list/matrix/structure of int, int between zero and one
        Extracted features in a format that can be used in the image classifier.
    """
    # <ASSIGNMENT: Implement your feature extraction by converting pixel intensities to features.>
    pixel_array = np.array(img) # Convert image data into array
    features = pixel_array/255 # Data normlization
    return features


def load_data_set(data_dir, n_validation = 0):
    """
    Prepare features for the images in data_dir and divide in a training and validation set.

    Parameters
    ----------
    data_dir : str
        Directory of images to load
    n_validation : int
        Number of images that are assigned to the validation set
    """

    # Extract png files
    files = os.listdir(data_dir)
    png_files = []
    for file in files:
        if file.split('.')[-1] == "png":
            png_files.append(file)

    random.shuffle(png_files)  # Shuffled list of the png-file names that are stored in data_dir

    # <ASSIGNMENT: Load the training and validation set and prepare the features and labels. Use extract_features()
    # to convert a loaded image (you can load an image with Image.open()) to features that can be processed by your
    # image classifier. You can extract the original label from the image filename.>
   
    # Creat parameter for later use
    data_features = []
    labels = []
    
    # Extra feature and label from the image in the png files
    for png in png_files:
        paths = os.path.join(data_dir, f'{png}') 
        data_feature = extract_features(Image.open(paths))
        label = png[0]
        data_features.append(data_feature)
        labels.append(label)

    # Encode labels
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)

    # In case n_validation = 0 to avoid getting errors that test split cannot be zero
    if n_validation == 0:
        training_features = data_features
        training_labels = labels
        validation_features = None
        validation_labels = None
    else:
        training_features, validation_features, training_labels, validation_labels = train_test_split(data_features,labels,test_size=n_validation)   

    return training_features, training_labels, validation_features, validation_labels


def generate_data_set(n_samples, data_dir):
    """
    Generate n_samples noisy images by using generate_noisy_image(), and store them in data_dir.

    Arguments
    ---------
    n_samples : int
        Number of train/test examples to generate
    data_dir : str in [TRAINING_IMAGE_DIR, TEST_IMAGE_DIR]
        Directory for storing images
    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)  # Generate a directory for data set storage, if not already present

    for i in range(n_samples):
        # <ASSIGNMENT: Replace with your implementation. Pick a random rank and convert it to a noisy image through
        # the generate_noisy_image() function below.>
        rank = random.choice(LABELS) # Randomly pick J,Q,K,A from LABEL 
        noise_level = random.choice(NOISE) # Assign random noise_level to image, the noise leve between 0 and 1 with 0.1 resolution.
        img = generate_noisy_image(rank, noise_level) # Generate noisy image

        img.save(f"{data_dir}\\{rank}_{i}.png")  # The filename encodes the original label for training/testing


def generate_noisy_image(rank, noise_level):
    """
    Generate a noisy image with a given noise corruption. This implementation mirrors how the server generates the
    images. However the exact server settings for noise_level and ROTATE_MAX_ANGLE are unknown.
    For the PokerBot assignment you won't need to update this function, but remember to test it.

    Arguments
    ---------
    rank : str in ['J', 'Q', 'K','A']
        Original card rank.
    noise_level : int between zero and one,
        Probability with which a given pixel is randomized.

    Returns
    -------
    noisy_img : Image
        A noisy image representation of the card rank.
    """

    if not 0 <= noise_level <= 1:
        raise ValueError(f"Invalid noise level: {noise_level}, value must be between zero and one")
    if rank not in LABELS:
        raise ValueError(f"Invalid card rank: {rank}")

    # Create rank image from text
    font = ImageFont.truetype(random.choice(FONTS), size = IMAGE_SIZE - 6)  # Pick a random font
    img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color = 255)
    draw = ImageDraw.Draw(img)
    (text_width, text_height) = draw.textsize(rank, font = font)  # Extract text size
    draw.text(((IMAGE_SIZE - text_width) / 2, (IMAGE_SIZE - text_height) / 2 - 4), rank, fill = 0, font = font)

    # Random rotate transformation
    img = img.rotate(random.uniform(-ROTATE_MAX_ANGLE, ROTATE_MAX_ANGLE), expand = False, fillcolor = '#FFFFFF')
    pixels = list(img.getdata())  # Extract image pixels

    # Introduce random noise
    for (i, _) in enumerate(pixels):
        if random.random() <= noise_level:
            pixels[i] = random.randint(0, 255)  # Replace a chosen pixel with a random intensity

    # Save noisy image
    noisy_img = Image.new('L', img.size)
    noisy_img.putdata(pixels)

    return noisy_img
