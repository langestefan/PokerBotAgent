import os
from data_sets import extract_features, load_data_set, generate_noisy_image
import numpy as np
import pytest


TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # Mark the test root directory
TRAINING_IMAGE_TEST_DIR = os.path.join(TEST_DIR, "data_sets", "training_images")
TEST_IMAGE_TEST_DIR = os.path.join(TEST_DIR, "data_sets", "test_images")


class TestDataSets:
    # <ASSIGNMENT: Test the extract_features(), load_data_set() and generate_noisy_image() functions in
    # data_sets.py. You can use the images under the test\data_sets\ directories for unit testing.
    # You don't need to test generate_data_set().>

    # Due to pre-defined data_sets is in wrong size (28 *28), so we replace the image in data_sets
    def test_extract_feature(self,image):
        '''
        Test extract_feature() function if it extract feature as expected
        '''
        # extract feature from the image
        test_feature = extract_features(image)
        # check the data if it is normalized, the value should in between 0.0 and 1.0
        assert np.max(test_feature) <= 1.0
        assert np.min(test_feature) >= 0.0

        # check the feature shape 
        assert test_feature.shape == (32,32)
        
    def test_load_data_set(self):
        '''
        Test load_data_set() function if it split traning sets and validation sets porperly
        '''
        # check when validation set is not required
        training_features, training_labels, validation_features, validation_labels = load_data_set(TRAINING_IMAGE_TEST_DIR)
        # check when validation = 0, validation set should return nothing
        assert validation_features == None and validation_labels == None

        # check training set
        assert len(training_labels) == 4 and len(training_features) == 4

        # check labels
        assert sorted(list(training_labels)) == [0, 1, 2, 3]
        assert training_labels.dtype == np.dtype('int64')

        # check when validation set is required
        training_features, training_labels, validation_features, validation_labels = load_data_set(TRAINING_IMAGE_TEST_DIR,1)
        # now validation set should contain 1 element
        assert len(validation_features) == 1 and len(validation_labels) == 1
        # Training set should contain 2 elements
        assert len(training_labels) == 3 and len(training_features) == 3
        
    
    
    def test_generate_noisy_image(self):
        '''
        Test generate_noisy_image() function if it generate image as required
        '''  
        # If noise level is out of the range, raises error
        assert pytest.raises(ValueError, generate_noisy_image, "K", 1.3)
        # If rank is out of the range, raises error
        assert pytest.raises(ValueError, generate_noisy_image, "5", 0.3)
        # Test whether the shape of the generated image meets the requirement
        generated_image_example = generate_noisy_image("Q", 0.5)
        assert generated_image_example.size == (32, 32)

       


